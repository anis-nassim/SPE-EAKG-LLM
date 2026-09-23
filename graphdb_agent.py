#!/usr/bin/env python3
"""
graphdb_agent.py — LLM agent that answers questions by querying a GraphDB
SPARQL endpoint through Gemini's function-calling interface.

The agent maintains a single, continuous conversation context across all
questions.  The LLM can autonomously issue SPARQL queries, inspect results,
self-correct on errors, and retry — up to a configurable cap per question.

Usage:
    python graphdb_agent.py [--questions questions.txt] \
                            [--prompt prompt_graphdb.txt] \
                            [--results results.txt] \
                            [--log log.json] \
                            [--model gemini-3.5-flash-lite]

Environment variables:
    GRAPHDB_ENDPOINT  URL of the local GraphDB repository
                      (e.g. http://localhost:7200/repositories/my-repo)
    GEMINI_API_KEY    Gemini API key
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import requests
from google import genai
from google.api_core import exceptions as api_exceptions
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3.5-flash-lite"

# Maximum number of tool calls the LLM may issue for a single question.
MAX_TOOL_CALLS_PER_QUESTION = 7

# Timeout (seconds) for every HTTP request to the SPARQL endpoint.
SPARQL_TIMEOUT_S = 10

# Retry parameters for Gemini rate-limit (429) errors.
MAX_RETRIES = 10
RETRY_COOLDOWN_S = 60

# Gemini request timeout (milliseconds).
GEMINI_REQUEST_TIMEOUT_MS = 120_000

# Regex that whitelists read-only SPARQL operations.
_READONLY_RE = re.compile(
    r"^\s*(SELECT|ASK|CONSTRUCT|DESCRIBE)\b", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# SPARQL endpoint configuration (resolved once at startup)
# ---------------------------------------------------------------------------

GRAPHDB_ENDPOINT: str = ""  # populated in main()

# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------


def run_sparql_query(query: str) -> str:
    """Execute a read-only SPARQL query against the GraphDB endpoint.

    Safety guards
    -------------
    * **Whitelist**: Only SELECT / ASK / CONSTRUCT / DESCRIBE are allowed.
    * **Timeout**: 10-second hard cap on the HTTP request.
    * **Error feedback**: Any exception message is returned verbatim so the
      LLM can diagnose the problem and rewrite the query.
    """

    # ---- 1. Whitelist check ------------------------------------------------
    if not _READONLY_RE.match(query):
        return (
            "ERROR — QUERY REJECTED: Only read-only SPARQL operations are "
            "allowed (SELECT, ASK, CONSTRUCT, DESCRIBE).  Your query appears "
            "to contain a write/update operation.  Please rewrite the query "
            "using a SELECT statement."
        )

    # ---- 2. Execute against GraphDB ----------------------------------------
    try:
        response = requests.post(
            GRAPHDB_ENDPOINT,
            data={"query": query},
            headers={"Accept": "application/sparql-results+json"},
            timeout=SPARQL_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return (
            "ERROR — TIMEOUT: The SPARQL query did not complete within "
            f"{SPARQL_TIMEOUT_S} seconds.  Consider simplifying the query, "
            "adding a LIMIT clause, or narrowing the graph pattern."
        )
    except requests.exceptions.ConnectionError as exc:
        return (
            f"ERROR — CONNECTION FAILED: Could not reach the SPARQL endpoint "
            f"at {GRAPHDB_ENDPOINT}.  Details: {exc}"
        )
    except requests.exceptions.HTTPError as exc:
        # GraphDB returns 400 for malformed SPARQL — surface the body.
        body = ""
        if exc.response is not None:
            body = exc.response.text[:2000]  # cap to avoid huge payloads
        return (
            f"ERROR — HTTP {exc.response.status_code if exc.response else '???'}: "
            f"{body or exc}"
        )
    except Exception as exc:
        return f"ERROR — UNEXPECTED: {type(exc).__name__}: {exc}"

    # ---- 3. Format and return results --------------------------------------
    try:
        data = response.json()
    except ValueError:
        # Non-JSON response (e.g. CONSTRUCT returning Turtle/RDF-XML).
        return response.text[:5000]

    # For SELECT queries, return a concise tabular summary.
    if "results" in data and "bindings" in data["results"]:
        bindings = data["results"]["bindings"]
        if not bindings:
            return "The query returned 0 results (empty result set)."

        variables = data.get("head", {}).get("vars", [])
        rows: list[str] = []
        for binding in bindings:
            row_parts = []
            for var in variables:
                cell = binding.get(var, {})
                row_parts.append(cell.get("value", ""))
            rows.append(" | ".join(row_parts))

        header = " | ".join(variables)
        table = f"{header}\n" + "-" * len(header) + "\n" + "\n".join(rows)
        return f"Results ({len(bindings)} rows):\n{table}"

    # For ASK queries:
    if "boolean" in data:
        return f"ASK result: {data['boolean']}"

    # Fallback: dump the raw JSON (truncated).
    return json.dumps(data, indent=2)[:5000]


# ---------------------------------------------------------------------------
# Gemini tool declaration
# ---------------------------------------------------------------------------

SPARQL_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_sparql_query",
            description=(
                "Execute a read-only SPARQL query against the GraphDB "
                "knowledge-graph endpoint and return the results.  "
                "Only SELECT, ASK, CONSTRUCT, and DESCRIBE queries are "
                "permitted.  If the query fails (syntax error, timeout, "
                "etc.), the error message is returned so you can fix and "
                "retry the query."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(
                        type=types.Type.STRING,
                        description="The SPARQL query string to execute.",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_text(path: str, label: str) -> str:
    """Read a file as UTF-8 text.  Exit with a clear message if missing."""
    p = pathlib.Path(path)
    if not p.is_file():
        print(f"Error: {label} file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def _parse_questions(text: str) -> list[str]:
    """Return non-empty, stripped lines from the questions text."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _send_with_retry(chat, message, *, is_function_response: bool = False):
    """Send a message to the Gemini chat, retrying on rate-limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chat.send_message(message)
        except Exception as exc:
            exc_text = str(exc)
            is_rate_limit = (
                "RESOURCE_EXHAUSTED" in exc_text or "429" in exc_text
            )
            if not is_rate_limit:
                raise  # non-retryable — let caller handle

            if attempt == MAX_RETRIES:
                raise  # exhausted retries

            print(
                f"  Rate-limited (attempt {attempt}/{MAX_RETRIES}). "
                f"Waiting {RETRY_COOLDOWN_S}s..."
            )
            for remaining in range(RETRY_COOLDOWN_S, 0, -1):
                print(f"   Retrying in {remaining}s...", end="\r")
                time.sleep(1)
            print("   Retrying now!                ")


# ---------------------------------------------------------------------------
# Agent loop for a single question
# ---------------------------------------------------------------------------


def _process_question(
    chat,
    question: str,
    question_index: int,
    total: int,
) -> dict:
    """Drive the tool-call loop for one question and return a log dict."""

    print(f"\n{'='*60}")
    print(f"  Question {question_index}/{total}")
    print(f"  {question}")
    print(f"{'='*60}")

    sparql_log: list[dict] = []
    tool_call_count = 0
    status = "completed"
    final_answer = ""

    # Send the question as the next user turn.
    response = _send_with_retry(chat, question)

    while True:
        # Check if the response contains a function call.
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None:
            final_answer = "(No response from the model.)"
            status = "error_no_candidate"
            break

        parts = candidate.content.parts if candidate.content else []

        # Collect any text parts as potential final answer.
        text_parts = [p.text for p in parts if p.text]

        # Look for function calls.
        fn_calls = [p.function_call for p in parts if p.function_call]

        if not fn_calls:
            # No tool call — the LLM has produced its final answer.
            final_answer = "\n".join(text_parts) if text_parts else "(Empty response)"
            break

        # Process each function call in the response.
        fn_responses: list[types.Part] = []
        for fc in fn_calls:
            if fc.name != "run_sparql_query":
                # Unknown function — tell the LLM.
                fn_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"error": f"Unknown function: {fc.name}"},
                    )
                )
                continue

            query = fc.args.get("query", "")
            tool_call_count += 1
            print(f"  [{tool_call_count}/{MAX_TOOL_CALLS_PER_QUESTION}] SPARQL ▶ {query[:120]}...")

            result = run_sparql_query(query)
            is_error = result.startswith("ERROR")

            sparql_log.append({
                "query": query,
                "result": result[:3000],  # cap for log readability
                "success": not is_error,
            })

            print(f"      {'✗' if is_error else '✓'} {result[:150]}...")

            fn_responses.append(
                types.Part.from_function_response(
                    name="run_sparql_query",
                    response={"result": result},
                )
            )

        # ---- Runaway cap check --------------------------------------------
        if tool_call_count >= MAX_TOOL_CALLS_PER_QUESTION:
            print(
                f"  ⚠ Runaway cap reached ({MAX_TOOL_CALLS_PER_QUESTION} tool calls). "
                f"Forcing final answer."
            )
            # Send the last tool result(s) together with a forcing instruction.
            force_msg = [
                *fn_responses,
                types.Part.from_text(
                    text="You have reached the maximum number of allowed tool calls "
                    "for this question.  You MUST now provide your best final "
                    "answer based on the data you have gathered so far.  Do NOT "
                    "call any more tools."
                ),
            ]
            response = _send_with_retry(chat, force_msg)

            # Extract the forced answer.
            forced_parts = (
                response.candidates[0].content.parts
                if response.candidates and response.candidates[0].content
                else []
            )
            final_answer = "\n".join(
                p.text for p in forced_parts if p.text
            ) or "(Model did not produce a final answer after cap.)"
            status = "capped"
            break

        # Send function responses back to the LLM so it can continue.
        response = _send_with_retry(chat, fn_responses)

    return {
        "question_index": question_index,
        "question": question,
        "sparql_queries": sparql_log,
        "final_answer": final_answer,
        "tool_calls_count": tool_call_count,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global GRAPHDB_ENDPOINT

    # ---- CLI arguments ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "LLM agent that answers questions by querying a GraphDB "
            "SPARQL endpoint via Gemini function calling."
        ),
    )
    parser.add_argument(
        "--questions",
        default="questions.txt",
        help="Path to the input file containing questions (one per line). "
             "(default: questions.txt)",
    )
    parser.add_argument(
        "--prompt",
        default="prompt_graphdb.txt",
        help="Path to the system prompt / instruction file. "
             "(default: prompt_graphdb.txt)",
    )
    parser.add_argument(
        "--results",
        default="results.txt",
        help="Path to the output file for LLM answers. "
             "(default: results.txt)",
    )
    parser.add_argument(
        "--log",
        default="log.json",
        help="Path to the JSON log file with detailed SPARQL query traces. "
             "(default: log.json)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    # ---- Environment variables --------------------------------------------
    GRAPHDB_ENDPOINT = os.getenv("GRAPHDB_ENDPOINT", "")
    if not GRAPHDB_ENDPOINT:
        print(
            "Error: GRAPHDB_ENDPOINT environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Read inputs ------------------------------------------------------
    prompt_text = _read_text(args.prompt, "System prompt")
    questions_text = _read_text(args.questions, "Questions")
    questions = _parse_questions(questions_text)

    if not questions:
        print("Error: No questions found in the questions file.", file=sys.stderr)
        sys.exit(1)

    # ---- Initialise Gemini client -----------------------------------------
    client = genai.Client(api_key=api_key)

    print(f"Model     : {args.model}")
    print(f"Endpoint  : {GRAPHDB_ENDPOINT}")
    print(f"Questions : {len(questions)}")
    print()

    # ---- Create chat session with tool + system instruction ----------------
    chat = client.chats.create(
        model=args.model,
        config=types.GenerateContentConfig(
            system_instruction=prompt_text,
            tools=[SPARQL_TOOL],
            http_options=types.HttpOptions(timeout=GEMINI_REQUEST_TIMEOUT_MS),
        ),
    )

    # ---- Prepare output files -----------------------------------------------
    results_path = pathlib.Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    log_path = pathlib.Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Process each question --------------------------------------------
    all_logs: list[dict] = []
    results_count = 0

    with results_path.open("w", encoding="utf-8") as fout:
        for idx, question in enumerate(questions, start=1):
            try:
                record = _process_question(chat, question, idx, len(questions))
            except Exception as exc:
                print(f"\n✗ Fatal error on question {idx}: {exc}", file=sys.stderr)
                record = {
                    "question_index": idx,
                    "question": question,
                    "sparql_queries": [],
                    "final_answer": f"FATAL ERROR: {exc}",
                    "tool_calls_count": 0,
                    "status": "error",
                }

                # Cool down before the next question to let rate limits reset.
                exc_text = str(exc)
                is_rate_limit = (
                    "RESOURCE_EXHAUSTED" in exc_text or "429" in exc_text
                )
                if is_rate_limit:
                    print(
                        f"  Rate-limit cooldown after fatal error. "
                        f"Waiting {RETRY_COOLDOWN_S}s before next question..."
                    )
                    for remaining in range(RETRY_COOLDOWN_S, 0, -1):
                        print(f"   Resuming in {remaining}s...", end="\r")
                        time.sleep(1)
                    print("   Resuming now!                ")

            # Write the LLM answer to the results file.
            fout.write(record["final_answer"] + "\n\n")
            fout.flush()
            results_count += 1

            # Accumulate the full record for the log file.
            all_logs.append(record)

            print(f"\n  ✓ Answer logged ({record['tool_calls_count']} tool calls, "
                  f"status={record['status']})")

    # ---- Write detailed JSON log -------------------------------------------
    log_path.write_text(
        json.dumps(all_logs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"Done — {results_count} answers written to {results_path}")
    print(f"       SPARQL query log written to {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
