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
Proxy eval of compression recipes vs BF16 for an OpenVINO LTX-2.x export (optimum-intel OVLTX2Pipeline,
distilled 8-step sigma schedule, all guidance off).

    python -P proxy_latent.py data <export_dir> --out work/proxy_data.pt
    python -P proxy_latent.py eval <export_dir> --data work/proxy_data.pt --candidate base=recipes/base.json ...

`data` runs the uncompressed (BF16 weights, f32 inference) pipeline on the proxy prompts with the distilled
recipe and records, per prompt: the text-encoder output fed to the connectors, the connector outputs, every
transformer call (inputs and video/audio velocity outputs) and the final video/audio latents.

`eval` compresses the components named in each recipe in memory (data-free nncf.compress_weights) and reports
the relative squared error vs the teacher, ||y - y_bf16||^2 / ||y_bf16||^2:
  tf_video / tf_audio   teacher-forced: the candidate transformer on the teacher's recorded inputs, averaged
                        over prompts and denoising steps (isolates the transformer's own error);
  fr_video / fr_audio   free-running: the whole candidate pipeline from the same noise, error of the final
                        latents (includes the text encoder and connectors, and error compounding over steps);
  fr_late_video         free-running velocity error over the last half of the steps;
  te / conn_video       free-running error of the text-encoder output and of the video connector output.
`--tf-only` skips the free-running part (transformer family screens); `--te-only` scores only the text encoder
and connectors on the recorded prompts. Teacher-forced results are cached per transformer recipe next to the data.

Inference runs in f32 with dynamic activation quantization off (OV_CONFIG), so the errors come from the
compressed weights only. For another pipeline, adapt load_pipe(), run_pipeline(), DISTILLED_KWARGS and the
Recorder hooks (which component calls to record).
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

DISTILLED_KWARGS = None  # filled in main(): needs diffusers
OV_CONFIG = {"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "0"}

PROMPTS = [
    (
        "A young man with short curly black hair and a denim jacket talks to the camera in a busy cafe, smiling and "
        "gesturing with his hands; warm afternoon light comes through the window behind him, and the background is "
        "softly blurred with people sitting at tables. The camera is handheld at eye level. Real-life footage."
    ),
    (
        "A golden retriever runs across a green meadow toward the camera, its ears flapping and tongue out, then "
        "jumps to catch a red frisbee in mid-air. Tall grass sways in the wind, and the sky is clear blue. The camera "
        "tracks the dog at a low angle in slow motion."
    ),
    (
        "A rainy city street at night: cars with glowing headlights drive past neon shop signs reflected in the wet "
        "asphalt, and pedestrians with umbrellas cross at a crosswalk. The camera slowly dollies forward along the "
        "sidewalk. Cinematic, shallow depth of field."
    ),
    (
        "Close-up of an elderly woman's hands kneading bread dough on a floured wooden table, then folding it and "
        "pressing it with her palms. Flour dust floats in a beam of morning sunlight. The camera is static, slightly "
        "above the table."
    ),
    (
        "A waterfall pours into a turquoise pool surrounded by mossy rocks and ferns in a tropical forest; mist rises "
        "and drifts across the frame, and a small bird flies through. The camera slowly pans from left to right."
    ),
    (
        "Two dancers in flowing red and white dresses spin and leap on a dark stage lit by a single spotlight; their "
        "skirts swirl as they move in unison. The camera circles around them at a medium distance. The scene appears "
        "to be from a stage performance."
    ),
]


def rel_err(a, b):
    a, b = a.double(), b.double()
    return ((a - b).pow(2).sum() / b.pow(2).sum().clamp_min(1e-30)).item()


def clone(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (list, tuple)):
        return type(x)(clone(v) for v in x)
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return x


class Recorder:
    """Wraps pipe.transformer.forward and pipe.connectors.forward to record their inputs and outputs."""

    def __init__(self, pipe, keep_inputs):
        self.keep_inputs, self.calls, self.connectors = keep_inputs, [], []
        tf, cn = pipe.transformer.forward, pipe.connectors.forward

        def transformer_forward(*args, **kwargs):
            out = tf(*args, **kwargs)
            self.calls.append({"inputs": clone(kwargs) if self.keep_inputs else None,
                               "video": clone(out[0]), "audio": clone(out[1])})  # fmt: skip
            return out

        def connectors_forward(*args, **kwargs):
            out = cn(*args, **kwargs)
            self.connectors.append({"text_embeds": clone(args[0] if args else kwargs["text_encoder_hidden_states"]),
                                    "out": clone(out)})  # fmt: skip
            return out

        pipe.transformer.forward = transformer_forward
        pipe.connectors.forward = connectors_forward

    def reset(self):
        self.calls, self.connectors = [], []


def run_pipeline(pipe, prompt, args):
    out = pipe(
        prompt=prompt, width=args.width, height=args.height, num_frames=args.num_frames,
        frame_rate=float(args.frame_rate), generator=torch.Generator().manual_seed(args.seed),
        output_type="latent", return_dict=False, **DISTILLED_KWARGS,
    )  # fmt: skip
    return out[0].detach().clone(), out[1].detach().clone()


def load_pipe(export_dir):
    from optimum.intel import OVLTX2Pipeline

    return OVLTX2Pipeline.from_pretrained(export_dir, load_in_8bit=False, compile=False, device="CPU",
                                          ov_config=dict(OV_CONFIG))  # fmt: skip


def build_data(args):
    pipe = load_pipe(args.export_dir)
    rec = Recorder(pipe, keep_inputs=True)
    prompts = PROMPTS[: args.num_prompts]
    data = {"settings": {k: getattr(args, k) for k in ("width", "height", "num_frames", "frame_rate", "seed")},
            "prompts": prompts, "samples": []}  # fmt: skip
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        rec.reset()
        video, audio = run_pipeline(pipe, prompt, args)
        conn = rec.connectors[0]
        data["samples"].append({"calls": rec.calls, "text_embeds": conn["text_embeds"], "connectors": conn["out"],
                                "video": video, "audio": audio})  # fmt: skip
        print(f"prompt {i}: {len(rec.calls)} transformer calls, video latents {tuple(video.shape)}, "
              f"audio latents {tuple(audio.shape)}, {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    torch.save(data, args.out)
    print(f"-> {args.out}")


def compress_pipe(pipe, recipe):
    sys.path.insert(0, str(Path(__file__).parent))
    from recipe_io import component_kwargs

    import nncf

    for comp, kw in recipe.items():
        if kw is None or comp.startswith("_"):
            continue
        part = pipe.components[comp]
        t0 = time.time()
        part.model = nncf.compress_weights(part.model, **component_kwargs(kw))
        part.request = None
        print(f"  compressed {comp} in {time.time() - t0:.0f}s", flush=True)


def evaluate(args):
    global OV_CONFIG
    data = torch.load(args.data, weights_only=False)
    for k, v in data["settings"].items():
        setattr(args, k, v)
    for cand in args.candidate:
        label, _, path = cand.partition("=")
        t0 = time.time()
        print(f"== {label}: {path}", flush=True)
        with open(path) as f:
            recipe = json.load(f)
        pipe = load_pipe(args.export_dir)
        if args.tf_only:
            compress_pipe(pipe, {"transformer": recipe.get("transformer")})
        elif args.te_only:
            compress_pipe(pipe, {k: recipe.get(k) for k in ("text_encoder", "connectors")})
        else:
            compress_pipe(pipe, recipe)
        t_compress = time.time() - t0

        if args.te_only:
            te, conn_v, conn_a = [], [], []
            for s, prompt in zip(data["samples"], data["prompts"]):
                pe, pm, _, _ = pipe.encode_prompt(prompt=prompt, do_classifier_free_guidance=False,
                                                  device=torch.device("cpu"))  # fmt: skip
                out = pipe.connectors(pe, pm, padding_side="left")
                te.append(rel_err(pe, s["text_embeds"]))
                conn_v.append(rel_err(out[0], s["connectors"][0]))
                conn_a.append(rel_err(out[1], s["connectors"][1]))
            result = {"label": label, "path": path, "te_only": True, "te": sum(te) / len(te),
                      "conn_video": sum(conn_v) / len(conn_v), "conn_audio": sum(conn_a) / len(conn_a),
                      "conn_video_per_prompt": conn_v, "total_s": round(time.time() - t0)}  # fmt: skip
            print(json.dumps(result), flush=True)
            with open(args.results, "a") as f:
                f.write(json.dumps(result) + "\n")
            del pipe
            gc.collect()
            continue

        # Teacher-forced transformer error; cached by the transformer's recipe (it does not depend on the rest).
        key = json.dumps([recipe.get("transformer"), args.tf_prompts], sort_keys=True)
        cache_path = Path(args.data).with_suffix(".tf_cache.jsonl")
        cache = {}
        if cache_path.exists():
            cache = {json.loads(line)["key"]: json.loads(line) for line in cache_path.read_text().splitlines()}
        if key in cache:
            tf_video, tf_audio, per_step_tf = cache[key]["tf_video"], cache[key]["tf_audio"], cache[key]["per_step"]
            print("  teacher-forced metrics from cache", flush=True)
        else:
            tf_video, tf_audio, per_step_tf = [], [], []
            for s in data["samples"][: args.tf_prompts]:
                steps = []
                for call in s["calls"]:
                    v, a = pipe.transformer.forward(**call["inputs"])
                    tf_video.append(rel_err(v, call["video"]))
                    tf_audio.append(rel_err(a, call["audio"]))
                    steps.append(tf_video[-1])
                per_step_tf.append(steps)
            with open(cache_path, "a") as f:
                entry = {"key": key, "tf_video": tf_video, "tf_audio": tf_audio, "per_step": per_step_tf}
                f.write(json.dumps(entry) + "\n")

        def mean(x):
            return sum(x) / len(x)

        if args.tf_only:
            result = {"label": label, "path": path, "tf_video": mean(tf_video), "tf_audio": mean(tf_audio),
                      "tf_video_per_step": [mean([p[i] for p in per_step_tf]) for i in range(len(per_step_tf[0]))],
                      "tf_only": True, "compress_s": round(t_compress), "total_s": round(time.time() - t0)}  # fmt: skip
            print(json.dumps(result), flush=True)
            with open(args.results, "a") as f:
                f.write(json.dumps(result) + "\n")
            del pipe
            gc.collect()
            continue

        # Free-running pipeline error.
        rec = Recorder(pipe, keep_inputs=False)
        fr_video, fr_audio, fr_late, te, conn_v, conn_a = [], [], [], [], [], []
        for s, prompt in zip(data["samples"][: args.num_prompts], data["prompts"]):
            rec.reset()
            video, audio = run_pipeline(pipe, prompt, args)
            fr_video.append(rel_err(video, s["video"]))
            fr_audio.append(rel_err(audio, s["audio"]))
            n = len(s["calls"])
            late = zip(rec.calls[n // 2 :], s["calls"][n // 2 :])
            fr_late.append(sum(rel_err(c["video"], t["video"]) for c, t in late) / (n - n // 2))
            te.append(rel_err(rec.connectors[0]["text_embeds"], s["text_embeds"]))
            conn_v.append(rel_err(rec.connectors[0]["out"][0], s["connectors"][0]))
            conn_a.append(rel_err(rec.connectors[0]["out"][1], s["connectors"][1]))

        result = {
            "label": label, "path": path,
            "tf_video": mean(tf_video), "tf_audio": mean(tf_audio),
            "fr_video": mean(fr_video), "fr_audio": mean(fr_audio), "fr_late_video": mean(fr_late),
            "te": mean(te), "conn_video": mean(conn_v), "conn_audio": mean(conn_a),
            "fr_video_per_prompt": fr_video,
            "tf_video_per_step": [mean([p[i] for p in per_step_tf]) for i in range(len(per_step_tf[0]))],
            "compress_s": round(t_compress), "total_s": round(time.time() - t0),
        }  # fmt: skip
        print(json.dumps(result), flush=True)
        with open(args.results, "a") as f:
            f.write(json.dumps(result) + "\n")
        del pipe, rec
        gc.collect()


def main():
    global DISTILLED_KWARGS
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("export_dir")
    d.add_argument("--out", default="proxy_data.pt")
    d.add_argument("--num-prompts", type=int, default=len(PROMPTS))
    d.add_argument("--width", type=int, default=768)
    d.add_argument("--height", type=int, default=448)
    d.add_argument("--num-frames", type=int, default=49)
    d.add_argument("--frame-rate", type=float, default=25.0)
    d.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("eval")
    e.add_argument("export_dir")
    e.add_argument("--data", required=True)
    e.add_argument("--candidate", action="append", required=True, help="LABEL=recipe.json")
    e.add_argument("--num-prompts", type=int, default=len(PROMPTS), help="prompts for the free-running metrics")
    e.add_argument("--tf-prompts", type=int, default=3, help="prompts for the teacher-forced metrics")
    e.add_argument("--tf-only", action="store_true", help="only the teacher-forced transformer metrics")
    e.add_argument("--te-only", action="store_true", help="only the text encoder + connectors output error")
    e.add_argument("--results", default="proxy_results.jsonl")
    args = ap.parse_args()

    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    DISTILLED_KWARGS = {
        "sigmas": DISTILLED_SIGMA_VALUES, "guidance_scale": 1.0, "audio_guidance_scale": 1.0, "stg_scale": 0.0,
        "audio_stg_scale": 0.0, "modality_scale": 1.0, "audio_modality_scale": 1.0,
    }  # fmt: skip
    build_data(args) if args.cmd == "data" else evaluate(args)


if __name__ == "__main__":
    main()
