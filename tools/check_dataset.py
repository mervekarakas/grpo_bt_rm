from grpo_bt_rm.data.summarize_from_feedback import load_comparisons_split, extract_pair

def main():
    train = load_comparisons_split("train")
    val = load_comparisons_split("validation")

    print("Train size:", len(train))
    print("Validation size:", len(val))

    ex = val[0]
    print("\nRaw keys:", ex.keys())

    post, s0, s1, label = extract_pair(ex)
    print("\nPost:\n", post[:500], "...\n")
    print("\nSummary 0:\n", s0[:300], "...\n")
    print("\nSummary 1:\n", s1[:300], "...\n")
    print("\nchoice label:", label)

if __name__ == "__main__":
    main()
