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
Table of proxy results with recipe sizes.

    python summarize.py work/results.jsonl [--base base] [--nodes-dir work] [--exclude-prompts 3]

fr_geo is the geometric mean over prompts of the free-running final-latent error: one chaotic prompt (where
even INT8 diverges from BF16) would otherwise dominate the arithmetic mean. fr_audio is the mean free-running
error of the audio latent.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from size import load_nodes  # noqa: E402
from size import recipe_size  # noqa: E402


def geo(xs):
    return math.exp(sum(math.log(max(x, 1e-12)) for x in xs) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--base", default="r1_base")
    ap.add_argument("--nodes-dir", default="work")
    ap.add_argument("--exclude-prompts", default="", help="comma-separated prompt indices to drop from fr_geo")
    args = ap.parse_args()

    comp_nodes = load_nodes(args.nodes_dir, ["transformer", "connectors", "text_encoder"])
    rows = {}
    for path in args.results:
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            rows[r["label"]] = r  # last result of a label wins
    drop = {int(i) for i in args.exclude_prompts.split(",") if i}
    base = rows.get(args.base)
    print(f"| recipe | GiB | tf_video | tf_audio | fr_geo | fr_audio | te | conn_video | tf_video vs {args.base} |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, r in rows.items():
        size = recipe_size(json.loads(Path(r["path"]).read_text()), comp_nodes)[0]
        fr = [x for i, x in enumerate(r.get("fr_video_per_prompt", [])) if i not in drop]
        rel = f"{r['tf_video'] / base['tf_video']:.2f}x" if base else ""
        frs = f"{geo(fr):.4f} | {r['fr_audio']:.4f}" if fr else "- | -"
        te = f"{r['te']:.1e} | {r['conn_video']:.1e}" if "te" in r else "- | -"
        print(f"| {label} | {size:.2f} | {r['tf_video']:.2e} | {r['tf_audio']:.2e} | {frs} | {te} | {rel} |")


if __name__ == "__main__":
    main()
