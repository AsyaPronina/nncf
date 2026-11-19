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
Predicted size of multi-component recipes and the precision each layer type gets, from arch.py's
nodes_<component>.json.

    python size.py base.json a.json ... --nodes-dir work [--components transformer connectors text_encoder]

A recipe is {"<component>": <nncf.compress_weights kwargs>, ...}; components missing from it stay BF16.
Stored bits per weight in OpenVINO (group g; -1 = per channel; zero points are u4/u8 per group):
    INT4_ASYM 4 + 20/g    INT4_SYM 4 + 16/g    INT8_ASYM 8 + 24/in    INT8_SYM 8 + 16/in    BF16 16
Without all_layers, NNCF puts embeddings (Gather) and the last MatMul in topological order into backup_mode.
ROWS groups LTX-2.x layer types; families of other models that match no row get a row of their own.
"""

import argparse
import collections
import json
import re
from pathlib import Path

BLOCK_RE = re.compile(r"(transformer_blocks|layers)\.(\d+)\.")


def family_of(node):
    name = node.split("/")[0].removeprefix("__module.").removeprefix("model.")
    return BLOCK_RE.sub(lambda m: f"{m.group(1)}.N.", name)


def select(selector, nodes):
    patterns = [re.compile(p) for p in selector.get("patterns", [])]
    return {
        n["node"]
        for n in nodes
        if n["node"] in selector.get("names", [])
        or any(p.match(n["node"]) for p in patterns)
        or n["node_type"] in selector.get("types", [])
    }


def bits(mode, group_size, node):
    if mode == "ignored":
        return 16.0
    g = node["in_features"] if group_size in (None, -1) else group_size
    return {
        "int2_sym": 2 + 16 / g, "int3_sym": 3 + 16 / g, "int4_asym": 4 + 20 / g, "int4_sym": 4 + 16 / g,
        "int8_asym": 8 + 24 / g, "int8_sym": 8 + 16 / g,
    }[mode]  # fmt: skip


def decisions(recipe, nodes):
    """{node: (mode, group_size)} with NNCF precedence: scoped configs beat ignored_scope; the last one wins."""
    if recipe is None:
        return {n["node"]: ("ignored", None) for n in nodes}
    mode = recipe["mode"]
    d = {n["node"]: (mode, recipe.get("group_size", 128 if mode.startswith("int4") else -1)) for n in nodes}
    if not recipe.get("all_layers") and not mode.startswith("int8"):
        backup = (recipe.get("backup_mode", "int8_asym"), -1)
        matmuls = [n for n in nodes if n["node_type"] == "MatMul"]
        for n in nodes:
            if n["node_type"] != "MatMul":
                d[n["node"]] = backup
        if matmuls:
            d[matmuls[-1]["node"]] = backup
    for name in select(recipe.get("ignored_scope", {}), nodes):
        d[name] = ("ignored", None)
    for sc in recipe.get("scoped_configs", []):
        for name in select(sc, nodes):
            d[name] = (sc["mode"], sc.get("group_size", -1))
    return d


def fmt_layers(layers):
    runs = []
    for x in sorted(layers):
        if runs and x == runs[-1][1] + 1:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)


def cell(choices):
    parts = []
    for (m, g), ns in sorted(choices.items(), key=lambda kv: -sum(n["num_weights"] for n in kv[1])):
        w = sum(n["num_weights"] for n in ns)
        avg = sum(bits(m, g, n) * n["num_weights"] for n in ns) / w
        if m == "ignored":
            text = "BF16"
        elif g in (None, -1):
            text = f"{m.upper()} ({avg:.2f})"
        else:
            text = f"{m.upper()} g{g} ({avg:.3f})"
        layers = {n["layer"] for n in ns if n["layer"] is not None}
        if len(choices) > 1 and layers:
            text += f" blk {fmt_layers(layers)}"
        parts.append(text)
    return "; ".join(parts)


# Rows of the table: layer types grouped the way the model is built.
ROWS = [
    ("video FF (ff.net.0/2)", r"^transformer_blocks\.N\.ff\."),
    ("video self-attn q/k (attn1)", r"^transformer_blocks\.N\.attn1\.to_[qk]$"),
    ("video self-attn v (attn1.to_v)", r"^transformer_blocks\.N\.attn1\.to_v$"),
    ("video self-attn out (attn1.to_out)", r"^transformer_blocks\.N\.attn1\.to_out"),
    ("video text cross-attn q (attn2.to_q)", r"^transformer_blocks\.N\.attn2\.to_q$"),
    ("video text cross-attn k/v (attn2)", r"^transformer_blocks\.N\.attn2\.to_[kv]$"),
    ("video text cross-attn out (attn2.to_out)", r"^transformer_blocks\.N\.attn2\.to_out"),
    ("audio->video attn (a2v q/k/v/out)", r"^transformer_blocks\.N\.audio_to_video_attn\.to_(q|k|v|out)"),
    ("video->audio attn (v2a q/k/v/out)", r"^transformer_blocks\.N\.video_to_audio_attn\.to_(q|k|v|out)"),
    ("audio FF", r"^transformer_blocks\.N\.audio_ff\."),
    ("audio self/cross attn (audio_attn1/2)", r"^transformer_blocks\.N\.audio_attn[12]\.to_(q|k|v|out)"),
    ("attn gate logits (all to_gate_logits)", r"^transformer_blocks\.N\..*to_gate_logits"),
    ("AdaLN / timestep embeddings", r"^(audio_)?(time_embed|prompt_adaln)|^av_cross_attn_"),
    ("proj_in / proj_out (video+audio)", r"^(audio_)?proj_(in|out)$"),
    ("connectors: text_proj_in (video+audio)", r"^(video|audio)_text_proj_in$"),
    ("connectors: gate logits", r"_connector\..*to_gate_logits"),
    ("connectors: video_connector blocks", r"^video_connector\."),
    ("connectors: audio_connector blocks", r"^audio_connector\."),
    ("text enc: mlp gate/up", r"^language_model\.layers\.N\.mlp\.(gate|up)_proj"),
    ("text enc: mlp down", r"^language_model\.layers\.N\.mlp\.down_proj"),
    ("text enc: attn q/k/v/o", r"^language_model\.layers\.N\.self_attn\."),
    ("text enc: embed_tokens", r"embed_tokens"),
]


def row_of(comp, family):
    for label, rx in ROWS:
        if re.search(rx, family):
            return label
    return f"{comp}: {family}"


def load_nodes(nodes_dir, comps):
    out = {}
    for comp in comps:
        with open(Path(nodes_dir) / f"nodes_{comp}.json") as f:
            arch = json.load(f)
        out[comp] = (arch["nodes"], sum(arch.get("uncompressed", {}).values()))
    return out


def recipe_size(recipe, comp_nodes):
    """(total GiB, {comp: GiB}, {comp: decisions})."""
    per, ds = {}, {}
    for comp, (nodes, uncompressed) in comp_nodes.items():
        d = decisions(recipe.get(comp), nodes)
        b = sum(bits(*d[n["node"]], n) * n["num_weights"] for n in nodes) + 16 * uncompressed
        per[comp], ds[comp] = b / 8 / 2**30, d
    return sum(per.values()), per, ds


def table(labels, results, comp_nodes):
    total_w = sum(sum(n["num_weights"] for n in nodes) + unc for nodes, unc in comp_nodes.values())
    rows = collections.OrderedDict()
    for comp, (nodes, _) in comp_nodes.items():
        for n in nodes:
            rows.setdefault(row_of(comp, family_of(n["node"])), []).append((comp, n))
    print("| Layer type | Share | " + " | ".join(labels) + " |")
    print("|---|---|" + "---|" * len(labels))
    for label, items in sorted(rows.items(), key=lambda kv: -sum(n["num_weights"] for _, n in kv[1])):
        cells = []
        for _, _, ds in results:
            choices = collections.defaultdict(list)
            for comp, n in items:
                choices[ds[comp][n["node"]]].append(n)
            cells.append(cell(choices))
        share = 100 * sum(n["num_weights"] for _, n in items) / total_w
        print(f"| {label} | {share:.2f}% | " + " | ".join(cells) + " |")
    for comp in comp_nodes:
        print(f"| = {comp} | | " + " | ".join(f"{per[comp]:.2f} GiB" for _, per, _ in results) + " |")
    print("| **Total** | | " + " | ".join(f"**{t:.2f} GiB**" for t, _, _ in results) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipes", nargs="+")
    ap.add_argument("--nodes-dir", default="work")
    ap.add_argument("--components", nargs="+", default=["transformer", "connectors", "text_encoder"])
    ap.add_argument("--no-table", action="store_true")
    args = ap.parse_args()

    comp_nodes = load_nodes(args.nodes_dir, args.components)
    results, labels = [], []
    base_total = None
    for path in args.recipes:
        with open(path) as f:
            recipe = json.load(f)
        res = recipe_size(recipe, comp_nodes)
        base_total = base_total or res[0]
        per = ", ".join(f"{c} {g:.2f}" for c, g in res[1].items())
        print(f"{Path(path).stem:28s} {res[0]:7.2f} GiB ({100 * (res[0] / base_total - 1):+5.1f}% vs first)  [{per}]")
        results.append(res)
        labels.append(Path(path).stem)
    if not args.no_table:
        print()
        table(labels, results, comp_nodes)


if __name__ == "__main__":
    main()
