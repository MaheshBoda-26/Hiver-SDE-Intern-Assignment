"""Interactive CLI labeling tool for customer support intent classification.

1. Loads eval/labeling_sample.csv
2. Imports INTENTS from src/constants.py
3. For each row, shows tweet_id and customer_text
4. Prompts: "Intent (number):" showing a numbered list of INTENTS to pick from
5. Prompts: "Escalate to human? (y/n):"
6. Prompts: "Reason (one line):"
7. Appends the labeled row to eval/golden_set.csv immediately after each entry
8. On startup, checks eval/golden_set.csv for already-labeled tweet_ids and skips them
9. Shows running count like "Labeled 12 / 300" at the top of each prompt
"""

import csv
import os
import sys

# Ensure src and root are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.constants import INTENTS
except ImportError:
    from constants import INTENTS

SAMPLE_PATH = os.path.join(PROJECT_ROOT, "eval", "labeling_sample.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "eval", "golden_set.csv")
COLUMNS = [
    "tweet_id",
    "customer_text",
    "brand_reply_text",
    "intent",
    "escalate",
    "escalate_reason",
]


def load_samples(path: str) -> list[dict]:
    """Load sample tweets from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample file not found: {path}")
    with open(path, mode="r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_labeled_ids(path: str) -> set[str]:
    """Inspect output file and return set of already labeled tweet_ids."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()

    with open(path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "tweet_id" in reader.fieldnames:
            return {row["tweet_id"] for row in reader if row.get("tweet_id")}

    # If the file exists with an incompatible/legacy schema, preserve it
    legacy_path = path.replace(".csv", "_legacy.csv")
    print(f"Notice: Existing '{path}' has legacy format. Renaming to '{legacy_path}'.")
    if os.path.exists(legacy_path):
        os.remove(legacy_path)
    os.rename(path, legacy_path)
    return set()


def append_row(path: str, row: dict) -> None:
    """Immediately append a labeled row to CSV."""
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def prompt_intent() -> str:
    """Prompt user for intent by index or name."""
    print("Available intents:")
    for i, intent in enumerate(INTENTS, 1):
        print(f"  {i}. {intent}")

    while True:
        choice = input("Intent (number): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(INTENTS):
                return INTENTS[idx - 1]
        elif choice in INTENTS:
            return choice
        print(f"Invalid input. Enter a number between 1 and {len(INTENTS)}.")


def prompt_escalate() -> str:
    """Prompt user for escalation decision."""
    while True:
        ans = input("Escalate to human? (y/n): ").strip().lower()
        if ans in ("y", "n"):
            return ans
        print("Please enter 'y' or 'n'.")


def prompt_reason() -> str:
    """Prompt user for one-line reason."""
    while True:
        reason = input("Reason (one line): ").strip()
        if reason:
            return reason
        print("Please provide a brief reason.")


def main() -> None:
    samples = load_samples(SAMPLE_PATH)
    total = len(samples)
    labeled_ids = get_labeled_ids(OUTPUT_PATH)

    unlabeled = [s for s in samples if s["tweet_id"] not in labeled_ids]
    done = total - len(unlabeled)

    if not unlabeled:
        print(f"\nAll {total} samples have already been labeled in {OUTPUT_PATH}!")
        return

    print(f"\nStarting labeling: {done} / {total} already labeled.")
    print("Press Ctrl+C anytime to pause. Progress is saved after every entry.\n")

    for row in unlabeled:
        print("=" * 60)
        print(f"Labeled {done} / {total}")
        print("-" * 60)
        print(f"tweet_id: {row['tweet_id']}")
        print(f"customer_text:\n{row['customer_text']}\n")

        intent = prompt_intent()
        escalate = prompt_escalate()
        reason = prompt_reason()

        record = {
            "tweet_id": row["tweet_id"],
            "customer_text": row["customer_text"],
            "brand_reply_text": row.get("brand_reply_text", ""),
            "intent": intent,
            "escalate": escalate,
            "escalate_reason": reason,
        }
        append_row(OUTPUT_PATH, record)
        done += 1
        print("Saved.\n")

    print("=" * 60)
    print(f"Completed! Labeled {done} / {total} samples -> {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\n\nSession paused. Progress is saved. Run again anytime to resume.")
        sys.exit(0)
