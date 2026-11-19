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
Per-tensor quantization types of a remote GGUF (HF Hub), read from its header only via HTTP range requests,
mapped onto the NNCF/OV node names of the LTX-2.5 transformer and connectors.

    python -I reference.py <repo_id> <file.gguf> \
        --nodes work/nodes_transformer.json work/nodes_connectors.json --out work/reference_<tag>.json

The parser is a plain-struct reader of the GGUF v3 header: no gguf package, no local copy of the file, and no
data from the file is executed; only the header is downloaded.
For another model, adapt to_diffusers() and RENAMES (original checkpoint names -> diffusers module names).
"""

import argparse
import collections
import json
import re
import struct

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K",
              11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
              19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4"}  # fmt: skip
SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

# Original LTX-2 checkpoint names -> diffusers names (from the diffusers LTX-2 conversion script).
RENAMES = [
    ("patchify_proj", "proj_in"), ("audio_patchify_proj", "audio_proj_in"),
    ("av_ca_video_scale_shift_adaln_single", "av_cross_attn_video_scale_shift"),
    ("av_ca_a2v_gate_adaln_single", "av_cross_attn_video_a2v_gate"),
    ("av_ca_audio_scale_shift_adaln_single", "av_cross_attn_audio_scale_shift"),
    ("av_ca_v2a_gate_adaln_single", "av_cross_attn_audio_v2a_gate"),
    ("audio_prompt_adaln_single", "audio_prompt_adaln"), ("prompt_adaln_single", "prompt_adaln"),
    ("video_embeddings_connector", "video_connector"), ("audio_embeddings_connector", "audio_connector"),
    ("transformer_1d_blocks", "transformer_blocks"),
    ("text_embedding_projection.video_aggregate_embed", "video_text_proj_in"),
    ("text_embedding_projection.audio_aggregate_embed", "audio_text_proj_in"),
]  # fmt: skip


class RangeReader:
    """Sequential reader over a remote file that fetches it in chunks with HTTP Range requests."""

    def __init__(self, url, chunk=8 << 20):
        self.url, self.chunk, self.buf, self.pos = url, chunk, b"", 0
        self.headers = build_hf_headers()

    def read(self, n):
        while len(self.buf) < self.pos + n:
            start = len(self.buf)
            r = requests.get(self.url, headers={**self.headers, "Range": f"bytes={start}-{start + self.chunk - 1}"},
                             timeout=120, allow_redirects=True)  # fmt: skip
            r.raise_for_status()
            self.buf += r.content
        out = self.buf[self.pos : self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self):
        return self.read(self.unpack("<Q")).decode("utf-8", errors="replace")

    def value(self, vtype):
        if vtype == 8:
            return self.string()
        if vtype == 9:
            itype, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(itype) for _ in range(n)]
        return self.unpack(SCALAR[vtype])


def read_header(url):
    f = RangeReader(url)
    if f.read(4) != b"GGUF":
        msg = "not a GGUF file"
        raise ValueError(msg)
    version, n_tensors, n_kv = f.unpack("<I"), f.unpack("<Q"), f.unpack("<Q")
    meta = {}
    for _ in range(n_kv):
        key = f.string()
        val = f.value(f.unpack("<I"))
        meta[key] = val if not (isinstance(val, list) and len(val) > 16) else f"<array of {len(val)}>"
    tensors = {}
    for _ in range(n_tensors):
        name = f.string()
        dims = [f.unpack("<Q") for _ in range(f.unpack("<I"))]
        ttype, _offset = f.unpack("<I"), f.unpack("<Q")
        tensors[name] = {"type": GGML_TYPES.get(ttype, str(ttype)), "shape": dims}
    return version, meta, tensors


def to_diffusers(name):
    name = name.removeprefix("model.diffusion_model.").removesuffix(".weight")
    if name.startswith("adaln_single."):
        name = "time_embed." + name.removeprefix("adaln_single.")
    elif name.startswith("audio_adaln_single."):
        name = "audio_time_embed." + name.removeprefix("audio_adaln_single.")
    for old, new in RENAMES:
        name = name.replace(old, new)
    return name


def family_of(name):
    return re.sub(r"(transformer_blocks|layers)\.(\d+)\.", r"\1.N.", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("filename")
    ap.add_argument("--nodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    version, meta, tensors = read_header(hf_hub_url(args.repo_id, args.filename))
    print(f"GGUF v{version}, {len(tensors)} tensors; arch={meta.get('general.architecture')}, "
          f"file_type={meta.get('general.file_type')}")  # fmt: skip
    by_name = {to_diffusers(k): v for k, v in tensors.items() if k.endswith(".weight")}

    result, unmapped = {}, []
    for path in args.nodes:
        with open(path) as fh:
            nodes = json.load(fh)["nodes"]
        for n in nodes:
            w = n["weight"].removeprefix("model.")
            if w in by_name:
                result[n["node"]] = by_name[w]["type"]
            else:
                unmapped.append(n["weight"])
    with open(args.out, "w") as fh:
        json.dump({"file": f"{args.repo_id}/{args.filename}", "meta": meta, "nodes": result,
                   "all_tensors": {k: v["type"] for k, v in tensors.items()}}, fh, indent=1)  # fmt: skip

    fams = collections.defaultdict(dict)
    for node, t in result.items():
        fams[family_of(node.split("/")[0].removeprefix("__module."))][node] = t
    for fam, per in sorted(fams.items()):
        common = collections.Counter(per.values()).most_common(1)[0][0]
        other = collections.defaultdict(list)
        for node, t in per.items():
            if t != common:
                m = re.search(r"(?:transformer_blocks|layers)\.(\d+)\.", node)
                other[t].append(int(m.group(1)) if m else -1)
        print(f"{fam:70s} {common:6s} {dict((k, sorted(v)) for k, v in other.items()) or ''}")
    print(f"mapped {len(result)} of {len(result) + len(unmapped)} NNCF weights -> {args.out}")
    if unmapped:
        print("not in the GGUF:", sorted({family_of(w) for w in unmapped}))
    types = collections.Counter(v["type"] for v in tensors.values())
    print("all tensor types in file:", dict(types))


if __name__ == "__main__":
    main()
