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
Inventory: the weights NNCF will compress in each component of a multi-component optimum-intel export.

    python -P arch.py <export_dir> --components transformer connectors text_encoder --out-dir work/

For every component writes nodes_<component>.json (node, weight, node_type, layer, shape, in_features,
num_weights + uncompressed families), the input of size.py and reference.py, and prints each family's share.
OV node names keep the block index (e.g. __module.model.transformer_blocks.12.attn1.to_q/aten::linear/MatMul),
so scoped configs can use per-layer patterns directly.
"""

import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import openvino as ov

import nncf
from nncf.common.factory import build_graph
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.algorithms.weight_compression.algorithm import get_weight_compression_configuration

BLOCK_RE = re.compile(r"(transformer_blocks|layers|blocks)\.(\d+)\.")


def layer_of(name):
    m = BLOCK_RE.search(name)
    return int(m.group(2)) if m else None


def family_of(name):
    """__module.model.transformer_blocks.12.attn1.to_q/aten::linear/MatMul -> model.transformer_blocks.N.attn1.to_q"""
    name = name.split("/")[0].removeprefix("__module.")
    return BLOCK_RE.sub(lambda m: f"{m.group(1)}.N.", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir")
    ap.add_argument("--components", nargs="+", default=["transformer", "connectors", "text_encoder"])
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    core = ov.Core()
    for comp in args.components:
        model = core.read_model(Path(args.export_dir) / comp / "openvino_model.xml")
        graph = build_graph(model)
        algo = WeightCompression(
            **get_weight_compression_configuration(mode=nncf.CompressWeightsMode.INT4_ASYM, ratio=1.0), subset_size=1
        )
        algo.set_backend_entity(model)
        params, _, _ = algo.get_weight_compression_parameters(model, graph)
        nodes = [
            {
                "node": wp.node_with_weight.node_name,
                "weight": wp.node_with_weight.node_name.split("/")[0].removeprefix("__module."),
                "weight_const": wp.weight_name,
                "node_type": wp.node_with_weight.node_type,
                "layer": layer_of(wp.node_with_weight.node_name),
                "shape": list(wp.weight_shape),
                "in_features": int(wp.weight_shape[wp.reduction_axes[0]]),
                "num_weights": int(wp.num_weights),
            }
            for wp in params
        ]
        compressed = {n["weight_const"] for n in nodes}
        uncompressed = collections.Counter()
        for op in model.get_ops():
            if op.get_type_name() != "Constant" or not op.get_element_type().is_real():
                continue
            if op.get_friendly_name() in compressed:
                continue
            size = int(np.prod(op.get_shape())) if len(op.get_shape()) else 1
            if size < 2:
                continue
            uncompressed[family_of(op.get_friendly_name())] += size
        out = Path(args.out_dir) / f"nodes_{comp}.json"
        with open(out, "w") as f:
            json.dump({"model": f"{args.export_dir}/{comp}", "nodes": nodes, "uncompressed": uncompressed}, f, indent=1)

        total = sum(n["num_weights"] for n in nodes)
        fams = collections.defaultdict(list)
        for n in nodes:
            fams[family_of(n["node"])].append(n)
        print(
            f"\n### {comp}: {len(nodes)} compressible weights, {total / 1e9:.3f}B params, "
            f"{sum(uncompressed.values()) / 1e6:.1f}M uncompressed float params -> {out}"
        )
        print(f"{'family':70s} {'count':>5s} {'shape':>16s} {'share':>7s}  node type")
        for fam, ns in sorted(fams.items(), key=lambda kv: -sum(n["num_weights"] for n in kv[1])):
            share = sum(n["num_weights"] for n in ns) / total
            print(f"{fam:70s} {len(ns):5d} {str(tuple(ns[0]['shape'])):>16s} {100 * share:6.2f}%  {ns[0]['node_type']}")


if __name__ == "__main__":
    main()
