# coding=utf-8
"""Run the mini set (D:/mini_asr_data) through the Rust `transcribe` binary.

The python-side counterpart is `D:/Qwen3-ASR/scripts/mini_eval.py`; this driver
shells out to `transcribe.exe` per clip and writes `hyps_<arm>.json` in the same
row shape (`id/dataset/config/lang/ref/hyp/lang_hyp/wer/cer`), plus per-clip
`elapsed_s`/`rtfx` so overall RTFx is computable.  One GPU job at a time — the
loop is serial.

Usage (conda env `myenv`):
  python scripts/mini_run.py --arm rust17int8 --model D:/Qwen3-ASR/models/Qwen3-ASR-1.7B-int8 \
      --out D:/mini_asr_data/results_rust --device vulkan --max-new 256
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

EXE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "target", "release", "transcribe.exe")


def parse_output(raw: str) -> tuple[str, str, float] | None:
    """(hyp, lang_hyp, elapsed_s) from one transcribe run's merged output."""
    lines = raw.replace("\r\n", "\n").split("\n")
    idx = None
    elapsed = None
    lang = ""
    for i, ln in enumerate(lines):
        m = re.search(r"elapsed=([\d.]+)s.*?lang=(\S*)", ln)
        if m:
            idx = i
            elapsed = float(m.group(1))
            lang = m.group(2)
    if idx is None or elapsed is None:
        return None
    hyp = "\n".join(lines[idx + 1:]).strip()
    return hyp, lang, elapsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/mini_asr_data")
    ap.add_argument("--arm", required=True, help="arm name, e.g. rust17int8")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="D:/mini_asr_data/results_rust")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--exe", default=EXE)
    ap.add_argument("--only", default="", help="comma list of dataset/config")
    args = ap.parse_args()

    import jiwer

    wer_t = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                           jiwer.Strip(), jiwer.ReduceToListOfListOfWords()])
    cer_t = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                           jiwer.Strip(), jiwer.ReduceToListOfListOfChars()])

    manifest = json.load(open(os.path.join(args.root, "manifest.json"), encoding="utf-8"))
    clips = manifest["clips"]
    if args.only:
        want = {s.strip() for s in args.only.split(",") if s.strip()}
        clips = [c for c in clips
                 if f"{c['dataset']}/{c.get('config', c['lang'])}" in want]
    total_dur = sum(c["dur"] for c in clips)
    print(f"[{args.arm}] clips: {len(clips)}  audio: {total_dur:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    failures = 0
    t_wall = time.time()
    t_inf = 0.0
    for i, c in enumerate(clips):
        wav = os.path.join(args.root, c["wav_rel"])
        cmd = [args.exe, "--model", args.model, "--wav", wav,
               "--max-new", str(args.max_new), "--device", args.device]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True)
        wall = time.time() - t0
        # stdout carries the elapsed line + transcript; stderr carries the
        # phase breakdown — keep them apart or the phases land in the hyp.
        parsed = parse_output(r.stdout.decode("utf-8", errors="replace"))
        if parsed is None:
            failures += 1
            err = r.stderr.decode("utf-8", errors="replace")[-400:]
            print(f"  !! clip {c['id']} failed (rc={r.returncode}) …{err}", flush=True)
            if failures > 5:
                print("too many failures, aborting", flush=True)
                return 1
            continue
        hyp, lang_hyp, elapsed = parsed
        t_inf += elapsed
        ref = c["ref"]
        rows.append({"id": c["id"], "dataset": c["dataset"],
                     "config": c.get("config", c["lang"]), "lang": c["lang"],
                     "ref": ref, "hyp": hyp, "lang_hyp": lang_hyp,
                     "wer": round(jiwer.wer(ref, hyp, reference_transform=wer_t,
                                            hypothesis_transform=wer_t), 4),
                     "cer": round(jiwer.wer(ref, hyp, reference_transform=cer_t,
                                            hypothesis_transform=cer_t), 4),
                     "elapsed_s": elapsed, "rtfx": round(c["dur"] / elapsed, 2)
                     if elapsed > 0 else 0.0,
                     "wall_s": round(wall, 2)})
        if (i + 1) % 20 == 0:
            print(f"[{args.arm}] {i+1}/{len(clips)}  inf {t_inf:.0f}s  wall {time.time()-t_wall:.0f}s",
                  flush=True)

    out = os.path.join(args.out, f"hyps_{args.arm}.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # per-group aggregate + overall
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['dataset']}/{r['config']}"].append(r)
    print(f"==== {args.arm}: {len(rows)} clips, failures {failures}, "
          f"audio {total_dur:.0f}s, inference {t_inf:.0f}s, "
          f"RTFx {total_dur / t_inf:.2f}, wall {time.time()-t_wall:.0f}s ====", flush=True)
    for g in sorted(groups):
        items = groups[g]
        mw = sum(r["wer"] for r in items) / len(items)
        mc = sum(r["cer"] for r in items) / len(items)
        print(f"  {g:24s} n={len(items):3d} WER={mw:.4f} CER={mc:.4f}", flush=True)
    mw = sum(r["wer"] for r in rows) / max(len(rows), 1)
    mc = sum(r["cer"] for r in rows) / max(len(rows), 1)
    print(f"  {'OVERALL (macro)':24s} n={len(rows):3d} WER={mw:.4f} CER={mc:.4f}", flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
