#!/usr/bin/env python3
import argparse
import csv
import os
import time
from pathlib import Path
from typing import List, Optional

def read_csv_table(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    cols = rows[0]
    data = rows[1:]
    return cols, data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Directory containing eval logs and summary files")
    ap.add_argument("--summary_csv", required=True, help="Path to summary.csv")
    ap.add_argument("--summary_md", default="", help="Optional path to summary.md")
    ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", ""), help="W&B project (or set WANDB_PROJECT)")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", ""), help="W&B entity/team (optional)")
    ap.add_argument("--run_name", default="", help="Optional run name")
    ap.add_argument("--group", default=os.environ.get("WANDB_GROUP", ""), help="Optional W&B group")
    ap.add_argument("--tags", default=os.environ.get("WANDB_TAGS", ""), help="Comma-separated tags (optional)")
    ap.add_argument("--include_logs", action="store_true", help="Also upload individual eval_*.log files as artifact")
    args = ap.parse_args()

    if not args.project:
        print("[wandb_log_eval_summary] WANDB_PROJECT not set; skipping W&B logging.")
        return

    # Import wandb lazily so the script can still run without it installed.
    import wandb  # noqa

    logdir = Path(args.logdir).resolve()
    csv_path = Path(args.summary_csv).resolve()
    md_path = Path(args.summary_md).resolve() if args.summary_md else None

    if not csv_path.exists():
        raise FileNotFoundError(f"summary_csv not found: {csv_path}")
    if md_path and not md_path.exists():
        print(f"[wandb_log_eval_summary] summary_md not found (skipping): {md_path}")
        md_path = None

    cols, data = read_csv_table(csv_path)

    # Reasonable default run name
    if not args.run_name:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = f"eval_summary_{logdir.name}_{stamp}"

    tags_list: List[str] = [t.strip() for t in (args.tags or "").split(",") if t.strip()]

    run = wandb.init(
        project=args.project,
        entity=args.entity or None,
        name=args.run_name,
        group=args.group or None,
        job_type="eval",
        tags=tags_list or None,
        config={
            "logdir": str(logdir),
            "summary_csv": str(csv_path),
            "summary_md": str(md_path) if md_path else "",
        },
    )

    # Log as a Table for convenient browsing
    if cols and data:
        table = wandb.Table(columns=cols, data=data)
        wandb.log({"eval/summary_table": table})
    else:
        print("[wandb_log_eval_summary] CSV seems empty; skipping Table logging.")

    # Upload as an artifact so you can download later
    art = wandb.Artifact(name=f"eval-summary-{logdir.name}", type="eval-summary")
    art.add_file(str(csv_path), name="summary.csv")
    if md_path:
        art.add_file(str(md_path), name="summary.md")

    if args.include_logs:
        for p in sorted(logdir.glob("eval_*.log")):
            # keep artifact size sane; skip gigantic logs if needed
            art.add_file(str(p), name=f"logs/{p.name}")

    run.log_artifact(art)
    run.finish()
    print("[wandb_log_eval_summary] Logged summary to W&B successfully.")

if __name__ == "__main__":
    main()
