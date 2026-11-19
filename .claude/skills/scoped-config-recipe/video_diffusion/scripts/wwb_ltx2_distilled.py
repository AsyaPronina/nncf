#!/usr/bin/env python3
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

r"""WWB reference and target runs for the distilled LTX-2.5 transformer.

The `wwb` CLI can only hand the pipeline a step count and a per-prompt video guidance scale.
The distilled transformer in Lightricks/LTX-2.5-Diffusers needs an explicit sigma schedule and
every guidance term switched off, so this script drives whowhatbench.Text2VideoEvaluator with
its own generation function instead.

  # 1. Reference videos from the PyTorch pipeline (saved next to gt.csv, in reference/)
  python wwb_ltx2_distilled.py --base-model Lightricks/LTX-2.5-Diffusers --gt-data ltx25_wwb/gt.csv

  # 2. Score an optimum-intel export against them
  python wwb_ltx2_distilled.py --target-model ./ltx25_ov --gt-data ltx25_wwb/gt.csv --output ltx25_wwb/ov

  # 2b. The same with OpenVINO runtime properties (a JSON string, or a path to a JSON file)
  python wwb_ltx2_distilled.py --target-model ./ltx25_ov --gt-data ltx25_wwb/gt.csv --output ltx25_wwb/ov_f32 \
      --device GPU --ov-config '{"INFERENCE_PRECISION_HINT": "f32"}'

Step 1 stores its prompts, size, frame count and seed in gt.settings.json. Step 2 reads them back,
so both sides are always sampled the same way. Only the video frames are scored; audio is dropped.
WWB fixes the frame rate at 25 fps for both generation and the saved .mp4 files.

Needs diffusers from the main branch (LTX-2.5 support), which is newer than the version WWB pins.
"""

import argparse
import gc
import json
import os
from importlib.resources import files

import torch
import whowhatbench
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

# Distilled recipe from the model card: fixed sigma schedule, no CFG / STG / modality guidance.
# Any scale left out here falls back to LTX2Pipeline's defaults, which are tuned for the full
# (non-distilled) transformer: 3.0, 7.0, 1.0, 1.0, 3.0, 3.0 in the order below.
DISTILLED_KWARGS = {
    "sigmas": DISTILLED_SIGMA_VALUES,
    "guidance_scale": 1.0,
    "audio_guidance_scale": 1.0,
    "stg_scale": 0.0,
    "audio_stg_scale": 0.0,
    "modality_scale": 1.0,
    "audio_modality_scale": 1.0,
}


def gen_video(model, prompt, width, height, num_frames, frame_rate, generator=None, **_fixed_by_recipe):
    """Generation hook for Text2VideoEvaluator.

    WWB also passes negative_prompt, num_inference_steps, guidance_scale, guidance_rescale and
    empty_adapters. They are ignored: the distilled recipe fixes the schedule and the guidance.
    """
    with torch.no_grad():
        output = model(
            prompt=prompt,
            width=int(width),
            height=int(height),
            num_frames=int(num_frames),
            frame_rate=float(frame_rate),
            generator=generator,
            **DISTILLED_KWARGS,
        )
    return output.frames[0]


def load_prompts(path, num_samples):
    if path:
        with open(path, encoding="utf-8") as f:
            items = json.load(f)
    else:
        # WWB's built-in text-to-video prompts; only the text is reused, not their sizes or guidance.
        builtin = files("whowhatbench").joinpath("prompts").joinpath("text_to_video_prompts.json")
        items = json.loads(builtin.read_text(encoding="utf-8"))
    prompts = [item["prompt"] if isinstance(item, dict) else item for item in items]
    return prompts[:num_samples] if num_samples else prompts


def build_test_data(settings):
    # Text2VideoEvaluator reads these columns by position, so their order matters.
    count = len(settings["prompts"])
    return {
        "prompt": settings["prompts"],
        "negative_prompt": [""] * count,
        "width": [settings["width"]] * count,
        "height": [settings["height"]] * count,
        "guidance_scale": [DISTILLED_KWARGS["guidance_scale"]] * count,
    }


def make_evaluator(gt_data, settings, base_model=None):
    return whowhatbench.Text2VideoEvaluator(
        base_model=base_model,
        gt_data=gt_data,
        test_data=build_test_data(settings),
        num_inference_steps=len(DISTILLED_SIGMA_VALUES),
        num_frames=settings["num_frames"],
        gen_video_fn=gen_video,
        seed=settings["seed"],
    )


def load_reference_pipeline(model_id, dtype, offload, vae_tiling):
    from diffusers import LTX2Pipeline

    pipe = LTX2Pipeline.from_pretrained(model_id, dtype=getattr(torch, dtype))
    if torch.cuda.is_available():
        if offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
    if vae_tiling:
        pipe.vae.enable_tiling()
    return pipe


def parse_ov_config(value):
    """Accepts what `wwb --ov-config` accepts: a path to a JSON file, or a JSON string."""
    if not value:
        return {}
    try:
        if os.path.isfile(value):
            with open(value, encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = json.loads(value)
    except json.JSONDecodeError as error:
        msg = f"--ov-config is neither a JSON file nor a JSON string: {error}"
        raise SystemExit(msg) from error
    if not isinstance(config, dict):
        msg = '--ov-config must be a JSON object, e.g. {"INFERENCE_PRECISION_HINT": "f32"}'
        raise SystemExit(msg)
    return config


def load_openvino_pipeline(model_dir, device, ov_config):
    from optimum.intel import OVPipelineForText2Video

    # optimum-intel compiles every component of the export (text encoder, transformer, VAE, ...) with ov_config.
    return OVPipelineForText2Video.from_pretrained(model_dir, device=device, ov_config=ov_config)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base-model", help="Diffusers model id or path. When given, the reference videos are (re)generated."
    )
    parser.add_argument("--target-model", help="Directory with an optimum-intel OpenVINO export to score.")
    parser.add_argument("--gt-data", required=True, help="CSV with the reference outputs, e.g. ltx25_wwb/gt.csv.")
    parser.add_argument("--output", default="wwb_output", help="Directory for target videos and metric CSVs.")
    parser.add_argument("--device", default="CPU", help="OpenVINO device for the target model.")
    parser.add_argument(
        "--ov-config",
        help="OpenVINO runtime properties for the target model: a path to a JSON file or a JSON string, "
        'e.g. \'{"INFERENCE_PRECISION_HINT": "f32"}\'. Not used by the PyTorch reference.',
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float32"], help="PyTorch dtype of the reference pipeline."
    )
    parser.add_argument(
        "--offload",
        default="model",
        choices=["model", "sequential", "none"],
        help="How the reference pipeline uses a CUDA GPU. model: one component on the GPU at a time, so the whole "
        "transformer must fit in VRAM. sequential: layer by layer, much slower, for smaller GPUs. "
        "none: the whole pipeline on the GPU. Without CUDA the pipeline runs on the CPU.",
    )
    parser.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Decode the reference videos in tiles to save memory. Can change frames near tile borders.",
    )
    settings = parser.add_argument_group(
        "sampling settings (reference run only; the target run reuses the stored ones)"
    )
    settings.add_argument(
        "--prompts", help="JSON file with a list of prompts. Default: WWB's built-in text-to-video prompts."
    )
    settings.add_argument("--num-samples", type=int, help="Use only the first N prompts.")
    settings.add_argument("--width", type=int, default=960)
    settings.add_argument("--height", type=int, default=544)
    settings.add_argument("--num-frames", type=int, default=121)
    settings.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.base_model and not args.target_model:
        parser.error("pass --base-model, --target-model, or both")
    return args


def main():
    args = parse_args()
    settings_path = os.path.splitext(args.gt_data)[0] + ".settings.json"
    # Parsed up front so a malformed config fails before any long generation starts.
    ov_config = parse_ov_config(args.ov_config)
    if ov_config and not args.target_model:
        print("Note: --ov-config only affects --target-model; the PyTorch reference does not use OpenVINO.")

    if args.base_model:
        if args.width % 32 or args.height % 32:
            msg = "--width and --height must be divisible by 32"
            raise SystemExit(msg)
        settings = {
            "prompts": load_prompts(args.prompts, args.num_samples),
            "width": args.width,
            "height": args.height,
            "num_frames": args.num_frames,
            "seed": args.seed,
            # Kept for the record only; the target run does not use it.
            "reference": {"model": args.base_model, "dtype": args.dtype, "vae_tiling": args.vae_tiling},
        }
        pipe = load_reference_pipeline(args.base_model, args.dtype, args.offload, args.vae_tiling)
        evaluator = make_evaluator(args.gt_data, settings, base_model=pipe)
        evaluator.dump_gt(args.gt_data)
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        print(f"Reference data: {args.gt_data}; sampling settings: {settings_path}")
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if not os.path.exists(settings_path):
            msg = f"{settings_path} not found. Collect the reference with --base-model first."
            raise SystemExit(msg)
        with open(settings_path, encoding="utf-8") as f:
            settings = json.load(f)
        evaluator = make_evaluator(args.gt_data, settings)

    if args.target_model:
        print(f"OpenVINO device: {args.device}; config: {ov_config}")
        target = load_openvino_pipeline(args.target_model, args.device, ov_config)
        # score() uses WWB's default generation function unless the hook is passed again here.
        per_prompt, summary = evaluator.score(target, gen_video, output_dir=args.output)
        os.makedirs(args.output, exist_ok=True)
        with open(os.path.join(args.output, "target_run.json"), "w", encoding="utf-8") as f:
            json.dump({"target_model": args.target_model, "device": args.device, "ov_config": ov_config}, f, indent=2)
        per_prompt.to_csv(os.path.join(args.output, "metrics_per_question.csv"))
        summary.to_csv(os.path.join(args.output, "metrics.csv"))
        evaluator.dump_predictions(os.path.join(args.output, "target.csv"))
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
