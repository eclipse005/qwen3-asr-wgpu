# coding=utf-8
"""Aggregate the four mini-set arms into one WER/CER/RTFx table.

Arms: hyps_<arm>.json written by scripts/mini_run.py (Rust) and by
mini_eval.py (python, via --out); RTFx for the python arms is parsed from
mini_eval's console log ("==== fp16 (Xs) ====").

Usage: python scripts/mini_report.py --out D:/mini_asr_data/results_rust
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

ARM_LABELS = {
    "py06fp16": "python-cuda 0.6B fp16 (original)",
    "py06int8": "python-cuda 0.6B int8 (quantized)",
    "py17fp16": "python-cuda 1.7B fp16 (original)",
    "py17int8": "python-cuda 1.7B int8 (quantized)",
    "wgpu06fp16": "wgpu-vulkan 0.6B fp16 (original)",
    "wgpu06int8": "wgpu-vulkan 0.6B INT8 (quantized)",
    "wgpu17fp16": "wgpu-vulkan 1.7B fp16 (original)",
    "wgpu17int8": "wgpu-vulkan 1.7B INT8 (quantized)",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/mini_asr_data/results_rust")
    ap.add_argument("--arms", default=",".join(ARM_LABELS))
    ap.add_argument("--root", default="D:/mini_asr_data")
    a = ap.parse_args()

    manifest = json.load(open(os.path.join(a.root, "manifest.json"), encoding="utf-8"))
    total_dur = sum(c["dur"] for c in manifest["clips"])

    report = {}
    for arm in a.arms.split(","):
        arm = arm.strip()
        p = os.path.join(a.out, f"hyps_{arm}.json")
        if not os.path.isfile(p):
            print(f"!! missing {p}")
            continue
        rows = json.load(open(p, encoding="utf-8"))
        groups = defaultdict(list)
        for r in rows:
            groups[f"{r['dataset']}/{r['config']}"].append(r)
        inf = sum(r.get("elapsed_s", 0) for r in rows)
        # python arms: elapsed comes from mini_eval's log if the driver didn't record it
        if inf == 0:
            log = os.path.join(a.out, f"log_{arm}.txt")
            if os.path.isfile(log):
                m = re.search(r"==== \S+ \((\d+(?:\.\d+)?)s\) ====",
                              open(log, encoding="utf-8", errors="replace").read())
                if m:
                    inf = float(m.group(1))
        report[arm] = {
            "groups": {g: (sum(r["wer"] for r in v) / len(v),
                           sum(r["cer"] for r in v) / len(v))
                       for g, v in groups.items()},
            "overall_wer": sum(r["wer"] for r in rows) / max(len(rows), 1),
            "overall_cer": sum(r["cer"] for r in rows) / max(len(rows), 1),
            "n": len(rows),
            "inf_s": inf,
            "rtfx": total_dur / inf if inf else 0.0,
        }

    all_groups = sorted({g for r in report.values() for g in r["groups"]})
    arms = [x.strip() for x in a.arms.split(",") if x.strip() in report]
    short = {x: ARM_LABELS[x].replace("python-cuda", "py").replace("wgpu-vulkan", "wgpu")
             .replace(" (original)", "").replace(" (quantized)", "").replace(" fp16", "-fp16")
             .replace(" INT8", "-INT8").replace(" int8", "-int8") for x in arms}
    print(f"{'group':22s} " + " ".join(f"{short[x]:>13s}" for x in arms))
    for g in all_groups:
        wcells, ccells = [], []
        for x in arms:
            w, c = report[x]["groups"].get(g, (float("nan"), float("nan")))
            wcells.append(f"{w:.4f}".rjust(13))
            ccells.append(f"{c:.4f}".rjust(13))
        print(f"{g + ' WER':22s} " + " ".join(wcells))
        print(f"{g + ' CER':22s} " + " ".join(ccells))
    print()
    for x in arms:
        r = report[x]
        print(f"{ARM_LABELS.get(x, x):34s} n={r['n']:3d}  macro WER {r['overall_wer']:.4f}  "
              f"CER {r['overall_cer']:.4f}  inference {r['inf_s']:.0f}s  RTFx {r['rtfx']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
