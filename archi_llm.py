#!/usr/bin/env python3
"""
archi_llm.py — Analyze an ArchiMate model file via a Gemini chat session.

Sends an initial prompt + model content as the first message, then asks each
question from a questions file as a follow-up turn in the same session.
All Q&A pairs are written to an output file.

Usage:
    python archi_llm.py [--input model] [--prompt-file prompt.txt] \
                        [--questions-file questions.txt] [--output result.txt] \
                        [--model gemini-2.5-flash]
"""

import argparse
import os
import sys
import pathlib
import time

from google import genai
from google.api_core import exceptions as api_exceptions
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate character-count ceiling for the first message (prompt + model).
# Gemini models have large context windows, but we guard against accidentally
# sending absurdly large payloads.
MAX_FIRST_MESSAGE_CHARS = 1_000_000

# Per-request timeout in seconds.  ArchiMate models can be large and answers
# detailed, so we allow a generous window.
REQUEST_TIMEOUT_S = 120

# How many times to retry a question after a 429 rate-limit error.
# On each retry, the script waits RETRY_COOLDOWN_S seconds before retrying.
MAX_RETRIES = 10
RETRY_COOLDOWN_S = 60

# Default model name — change as newer versions are released.
DEFAULT_MODEL = "gemini-3.5-flash-lite"
 
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- CLI arguments ----------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Analyze an ArchiMate model file by chatting with Gemini.",
    )
    parser.add_argument(
        "--input",
        default="model",
        help="Path to the ArchiMate model file (default: model)",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Path to the initial/instruction prompt file (default: prompt.txt)",
    )
    parser.add_argument(
        "--questions-file",
        default="questions.txt",
        help="Path to the questions file, one per line (default: questions.txt)",
    )
    parser.add_argument(
        "--output",
        default="result.txt",
        help="Path for the output Q&A file (default: result.txt)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    # ---- API key ----------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY environment variable is not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Read inputs ------------------------------------------------------
    model_content = _read_text(args.input, "Input model")
    prompt_text = _read_text(args.prompt_file, "Prompt")
    questions_text = _read_text(args.questions_file, "Questions")

    questions = _parse_questions(questions_text)
    if not questions:
        print("Error: No questions found in the questions file.", file=sys.stderr)
        sys.exit(1)

    # ---- Size guard -------------------------------------------------------
    first_message = (
        f"{prompt_text}\n\n"
        f"--- MODEL CONTENT START ---\n"
        f"{model_content}\n"
        f"--- MODEL CONTENT END ---"
    )

    if len(first_message) > MAX_FIRST_MESSAGE_CHARS:
        print(
            f"Warning: The initial message is {len(first_message):,} characters, "
            f"which exceeds the {MAX_FIRST_MESSAGE_CHARS:,}-character safety "
            f"threshold.  Aborting to avoid excessive token usage.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Initialise Gemini client -----------------------------------------
    client = genai.Client(api_key=api_key)

    print(f"Model : {args.model}")
    print(f"Input : {args.input} ({len(model_content):,} chars)")
    print(f"Questions: {len(questions)}")
    print()

    # ---- Start chat session -----------------------------------------------
    chat = client.chats.create(
        model=args.model,
        config=types.GenerateContentConfig(
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000),
        ),
    )

    # Send the initial prompt + model content as the first turn.
    print("Sending initial prompt + model content...")
    try:
        response = chat.send_message(first_message)
    except api_exceptions.ResourceExhausted as exc:
        print(
            f"Rate-limit / quota error on initial message: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)
    except Exception as exc:
        print(f"Error sending initial message: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Model acknowledged ({len(response.text):,} chars). Starting Q&A.\n")

    # ---- Prepare output file ----------------------------------------------
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Ask each question ------------------------------------------------
    # Results are written incrementally so partial progress is never lost.
    results_count = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, question in enumerate(questions, start=1):


            print(f"Sending question {idx}/{len(questions)}...")

            # Retry loop: on 429 rate-limit errors, wait and retry.
            answer = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    answer = chat.send_message(question)
                    break  # success
                except Exception as exc:
                    exc_text = str(exc)
                    is_rate_limit = (
                        "RESOURCE_EXHAUSTED" in exc_text or "429" in exc_text
                    )

                    if not is_rate_limit:
                        # Non-retryable error — abort.
                        print(
                            f"\nError on question {idx}: {exc}",
                            file=sys.stderr,
                        )
                        print(f"{results_count} answers saved to {output_path} before failure.")
                        sys.exit(2)

                    if attempt == MAX_RETRIES:
                        print(
                            f"\nRate-limit error on question {idx} after "
                            f"{MAX_RETRIES} retries: {exc}",
                            file=sys.stderr,
                        )
                        print(f"{results_count} answers saved to {output_path} before failure.")
                        sys.exit(2)

                    print(
                        f"  Rate-limited on question {idx} (attempt {attempt}/{MAX_RETRIES}). "
                        f"Waiting {RETRY_COOLDOWN_S}s before retry..."
                    )
                    for remaining in range(RETRY_COOLDOWN_S, 0, -1):
                        print(f"   Retrying in {remaining}s...", end="\r")
                        time.sleep(1)
                    print("   Retrying now!                ")

            # Write this answer immediately so progress is never lost.
            fout.write(f"Q{idx}: {question}\n")
            fout.write(f"A{idx}: {answer.text}\n\n")
            fout.flush()
            results_count += 1

    print(f"\nDone — {results_count} answers written to {output_path}")


if __name__ == "__main__":
    main()
