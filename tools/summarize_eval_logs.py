#!/usr/bin/env python3
import argparse, glob, os, re, csv
import numpy as np

# --- basic metrics ---
RE_ACC  = re.compile(r"Accuracy \(ties=0.5\):\s*([0-9.]+)")
RE_NT   = re.compile(r"Non-tie accuracy:\s*([0-9.]+)")
RE_TIE  = re.compile(r"Tie rate:\s*([0-9.]+)")
RE_BT_A = re.compile(r"BT log-loss \(all pairs\): mean=([0-9.]+)")
RE_BT_N = re.compile(r"BT log-loss \(non-tie\):\s*mean=([0-9.]+)")

# per-run file naming (default)
RE_CKPT = re.compile(r"eval_(checkpoint-\d+)_seed(\d+)\.log")

def mk_diag_re(prefix: str, which: str):
    # matches lines like:
    # delta_pref (wrong): n=... mean=... std=... ... p5=... p50=... p95=...
    # bt_loss (wrong):    n=... mean=... std=... ... p5=... p50=... p95=...
    return re.compile(
        rf"{re.escape(prefix)} \({re.escape(which)}\): n=\d+ "
        rf"mean=([-0-9.]+) std=([0-9.]+).*?"
        rf"p5=([-0-9.]+).*?p50=([-0-9.]+).*?p95=([-0-9.]+)",
        re.S
    )

RE_D_ALL = mk_diag_re("delta_pref", "all non-tie")
RE_D_COR = mk_diag_re("delta_pref", "correct")
RE_D_WRO = mk_diag_re("delta_pref", "wrong")

RE_BT_ALLNT = mk_diag_re("bt_loss", "all non-tie")
RE_BT_COR   = mk_diag_re("bt_loss", "correct")
RE_BT_WRO   = mk_diag_re("bt_loss", "wrong")

def grab1(txt, r):
    m = r.search(txt)
    return float(m.group(1)) if m else np.nan

def grab_diag(txt, r):
    m = r.search(txt)
    if not m:
        return dict(mean=np.nan, std=np.nan, p5=np.nan, p50=np.nan, p95=np.nan)
    return dict(
        mean=float(m.group(1)),
        std=float(m.group(2)),
        p5=float(m.group(3)),
        p50=float(m.group(4)),
        p95=float(m.group(5)),
    )

def ms(xs):
    xs = [x for x in xs if not np.isnan(x)]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (np.nan, np.nan)

def fmt(x):
    return "nan" if np.isnan(x) else f"{x:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True, help="Directory containing eval_*.log files")
    ap.add_argument("--pattern", default="eval_checkpoint-*_seed*.log",
                    help="Glob pattern under logdir")
    ap.add_argument("--name_re", default="eval_(checkpoint-\\d+)_seed(\\d+)\\.log",
                    help="Regex to extract ckpt/seed from filename (2 groups)")
    ap.add_argument("--out_md", default="", help="Write markdown table to this file")
    ap.add_argument("--out_csv", default="", help="Write CSV to this file")
    args = ap.parse_args()

    re_name = re.compile(args.name_re)

    logs = sorted(glob.glob(os.path.join(args.logdir, args.pattern)))
    if not logs:
        raise SystemExit(f"No logs matched: {os.path.join(args.logdir, args.pattern)}")

    data = {}
    for p in logs:
        m = re_name.search(os.path.basename(p))
        if not m:
            continue
        ckpt, seed = m.group(1), int(m.group(2))
        txt = open(p, "r", encoding="utf-8", errors="ignore").read()

        row = dict(
            seed=seed,
            acc=grab1(txt, RE_ACC),
            nt=grab1(txt, RE_NT),
            tie=grab1(txt, RE_TIE),
            bt_all=grab1(txt, RE_BT_A),
            bt_nt=grab1(txt, RE_BT_N),
            d_all=grab_diag(txt, RE_D_ALL),
            d_cor=grab_diag(txt, RE_D_COR),
            d_wro=grab_diag(txt, RE_D_WRO),
            bt_allnt_diag=grab_diag(txt, RE_BT_ALLNT),
            bt_cor_diag=grab_diag(txt, RE_BT_COR),
            bt_wro_diag=grab_diag(txt, RE_BT_WRO),
        )
        data.setdefault(ckpt, []).append(row)

    cols = [
        "ckpt",
        "acc_mean","acc_std",
        "nt_mean","nt_std",
        "tie_mean","tie_std",
        "bt_nt_mean","bt_nt_std",
        "d_wro_mean","d_wro_std","d_wro_p5","d_wro_p50","d_wro_p95",
        "bt_wro_mean","bt_wro_std","bt_wro_p50","bt_wro_p95",
        "n",
    ]

    lines = []
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"]*len(cols)) + "|"
    lines.append(header)
    lines.append(sep)

    rows_out = []

    for ckpt in sorted(data.keys(), key=lambda x: int(x.split("-")[1])):
        rows = sorted(data[ckpt], key=lambda r: r["seed"])

        am, asd = ms([r["acc"] for r in rows])
        nm, nsd = ms([r["nt"] for r in rows])
        tm, tsd = ms([r["tie"] for r in rows])
        bNm, bNsd = ms([r["bt_nt"] for r in rows])

        d_wro_mean_m, d_wro_mean_sd = ms([r["d_wro"]["mean"] for r in rows])
        d_wro_std_m,  d_wro_std_sd  = ms([r["d_wro"]["std"]  for r in rows])
        d_wro_p5_m,   _             = ms([r["d_wro"]["p5"]   for r in rows])
        d_wro_p50_m,  _             = ms([r["d_wro"]["p50"]  for r in rows])
        d_wro_p95_m,  _             = ms([r["d_wro"]["p95"]  for r in rows])

        bt_wro_mean_m, bt_wro_mean_sd = ms([r["bt_wro_diag"]["mean"] for r in rows])
        bt_wro_std_m,  bt_wro_std_sd  = ms([r["bt_wro_diag"]["std"]  for r in rows])
        bt_wro_p50_m,  _              = ms([r["bt_wro_diag"]["p50"]  for r in rows])
        bt_wro_p95_m,  _              = ms([r["bt_wro_diag"]["p95"]  for r in rows])

        row_md = (
            f"| {ckpt} | {fmt(am)} | {fmt(asd)} | {fmt(nm)} | {fmt(nsd)} | {fmt(tm)} | {fmt(tsd)} | "
            f"{fmt(bNm)} | {fmt(bNsd)} | "
            f"{fmt(d_wro_mean_m)} | {fmt(d_wro_mean_sd)} | {fmt(d_wro_p5_m)} | {fmt(d_wro_p50_m)} | {fmt(d_wro_p95_m)} | "
            f"{fmt(bt_wro_mean_m)} | {fmt(bt_wro_mean_sd)} | {fmt(bt_wro_p50_m)} | {fmt(bt_wro_p95_m)} | "
            f"{len(rows)} |"
        )
        lines.append(row_md)

        rows_out.append(dict(
            ckpt=ckpt,
            acc_mean=am, acc_std=asd,
            nt_mean=nm, nt_std=nsd,
            tie_mean=tm, tie_std=tsd,
            bt_nt_mean=bNm, bt_nt_std=bNsd,
            d_wro_mean=d_wro_mean_m, d_wro_std=d_wro_mean_sd,
            d_wro_p5=d_wro_p5_m, d_wro_p50=d_wro_p50_m, d_wro_p95=d_wro_p95_m,
            bt_wro_mean=bt_wro_mean_m, bt_wro_std=bt_wro_mean_sd,
            bt_wro_p50=bt_wro_p50_m, bt_wro_p95=bt_wro_p95_m,
            n=len(rows),
        ))

    table = "\n".join(lines)
    print(table)

    if args.out_md:
        os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(table + "\n")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            for r in rows_out:
                w.writerow(r)

if __name__ == "__main__":
    main()
