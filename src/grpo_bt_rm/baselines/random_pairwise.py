import argparse
import random
from grpo_bt_rm.data.summarize_from_feedback import load_comparisons_split, extract_pair

def random_baseline_accuracy(split: str, n_examples: int = 1000, seed: int = 0) -> float:
    data = load_comparisons_split(split)
    rng = random.Random(seed)
    correct = 0
    total = 0

    for i in range(min(n_examples, len(data))):
        ex = data[i]
        _, _, _, label = extract_pair(ex)
        pred = rng.randint(0, 1)
        correct += int(pred == label)
        total += 1

    return correct / total if total else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n_examples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc = random_baseline_accuracy(args.split, args.n_examples, args.seed)
    print(f"Random baseline accuracy on {args.n_examples} {args.split} pairs: {acc:.3f}")

if __name__ == "__main__":
    main()
