# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
WWB summary table: mean metrics per run, and the paired difference of each metric vs a reference run
(Δ = run - ref; higher is better for similarity/SSIM, lower is better for LPIPS/tLPIPS).

    python wwb_compare.py --ref base [--wwb-dir wwb] [--extra "OV BF16=path/to/wwb_output" ...]

Every subdirectory of --wwb-dir with a metrics_per_question.csv is a run (run_wwb.sh writes them there).
"""

import argparse
import contextlib
import csv
import math
import sys
from pathlib import Path


def load(d):
    with open(Path(d) / "metrics_per_question.csv") as f:
        rows = list(csv.DictReader(f))
    keys = [k for k in rows[0] if k and k != "prompt" and not k.startswith("Unnamed")]
    out = {}
    for k in keys:
        with contextlib.suppress(ValueError):  # non-numeric columns
            out[k] = [float(r[k]) for r in rows]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="run name to compute paired differences against")
    ap.add_argument("--wwb-dir", default="wwb")
    ap.add_argument("--extra", action="append", default=[], help="LABEL=DIR: another WWB output dir to include")
    args = ap.parse_args()

    runs = dict(e.split("=", 1) for e in args.extra)
    for d in sorted(Path(args.wwb_dir).iterdir()):
        if (d / "metrics_per_question.csv").exists():
            runs[d.name] = str(d)
    data = {k: load(v) for k, v in runs.items() if (Path(v) / "metrics_per_question.csv").exists()}
    ref = data.get(args.ref)
    keys = ("similarity", "SSIM (higher is better, 1.0 best)", "LPIPS (lower is better, 0 best)",
            "tLPIPS (lower is better, 0 best)")  # fmt: skip
    short = ("sim", "SSIM", "LPIPS", "tLPIPS")
    print("| run | similarity | SSIM | LPIPS | tLPIPS | " + " | ".join(f"Δ{k} (SE)" for k in short) + " |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name, m in data.items():
        means = " | ".join(f"{sum(m[k]) / len(m[k]):.4f}" if k in m else "-" for k in keys)
        diffs = []
        for k in keys:
            if not ref or name == args.ref or k not in m or k not in ref:
                diffs.append("")
                continue
            d = [a - b for a, b in zip(m[k], ref[k])]
            mu = sum(d) / len(d)
            se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d))
            diffs.append(f"{mu:+.4f} ({se:.4f})")
        print(f"| {name} | {means} | {' | '.join(diffs)} |")


if __name__ == "__main__":
    sys.exit(main())
