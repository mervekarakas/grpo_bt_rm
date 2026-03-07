"""Convert Anthropic/hh-rlhf to ms-swift RM training format (JSONL).

Output format (per line):
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "chosen"}],
  "rejected_messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "rejected"}]
}

Mirrors ms-swift's built-in HHRLHFPreprocessor logic.
"""
import json
import re
import sys
from datasets import load_dataset


def _to_messages(parts):
    """Convert alternating [user, assistant, user, assistant, ...] parts to messages."""
    messages = []
    for i in range(0, len(parts) - 1, 2):
        messages.append({"role": "user", "content": parts[i]})
        messages.append({"role": "assistant", "content": parts[i + 1]})
    # Handle odd trailing part (shouldn't happen in well-formed data)
    if len(parts) % 2 == 1:
        messages.append({"role": "user", "content": parts[-1]})
    return messages


def convert_row(row):
    chosen = row["chosen"].strip()
    rejected = row["rejected"].strip()

    parts_chosen = [s.strip() for s in re.split(r"\n\nHuman:|\n\nAssistant:|\n\nHum:", chosen)]
    parts_rejected = [s.strip() for s in re.split(r"\n\nHuman:|\n\nAssistant:|\n\nHum:", rejected)]

    # Remove empty leading element
    parts_chosen = [p for p in parts_chosen if p]
    parts_rejected = [p for p in parts_rejected if p]

    if not parts_chosen or not parts_rejected:
        return None

    messages = _to_messages(parts_chosen)
    rejected_messages = _to_messages(parts_rejected)

    # Validate: both should have at least 1 user + 1 assistant turn
    if len(messages) < 2 or len(rejected_messages) < 2:
        return None

    return {"messages": messages, "rejected_messages": rejected_messages}


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    output = sys.argv[2] if len(sys.argv) > 2 else f"/data/mkarakas/experiments/grpo_bt_rm/bt_rm_data/hh_{split}.jsonl"

    print(f"Loading Anthropic/hh-rlhf split={split}...")
    ds = load_dataset("Anthropic/hh-rlhf")
    data = ds[split]
    print(f"Loaded {len(data)} examples")

    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)

    n_written = 0
    n_skipped = 0
    with open(output, "w") as f:
        for row in data:
            result = convert_row(row)
            if result is None:
                n_skipped += 1
                continue
            f.write(json.dumps(result) + "\n")
            n_written += 1

    print(f"Written {n_written} examples to {output} (skipped {n_skipped})")


if __name__ == "__main__":
    main()
