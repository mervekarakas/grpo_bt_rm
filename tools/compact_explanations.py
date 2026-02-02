import json
import argparse
import os

def compact_file(input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    n_in = 0
    n_out = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            rec = json.loads(line)

            # robustly grab fields from the verbose format
            post = rec.get("post", "")
            summary_pref = rec.get("summary_pref", "")
            summary_nonpref = rec.get("summary_nonpref", "")
            explanation = rec.get("explanation", "")
            idx = rec.get("idx", rec.get("id", n_in))

            compact = {
                "idx": int(idx),
                "post": post,
                "summary_pref": summary_pref,
                "summary_nonpref": summary_nonpref,
                "explanation": explanation,
            }
            fout.write(json.dumps(compact, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Read {n_in} records from {input_path}, wrote {n_out} compact records to {output_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="Path to original verbose JSONL file.")
    ap.add_argument("--output", type=str, required=True, help="Path to compact JSONL file.")
    return ap.parse_args()


def main():
    args = parse_args()
    compact_file(args.input, args.output)


if __name__ == "__main__":
    main()
