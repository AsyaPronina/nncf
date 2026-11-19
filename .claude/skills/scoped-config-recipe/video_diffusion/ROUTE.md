# Scoped-config recipe: video-diffusion route

For an OpenVINO export (optimum-intel) of a multi-component video-diffusion pipeline. Built and measured on
LTX-2.5 distilled (text to video and audio) on CPU, with three compressed components:

- `transformer`: 19B parameters
- `connectors`: 3.2B
- `text_encoder`: Gemma 4, 12B

This route follows the method in `SKILL.md`. Priors per weight family and the measured LTX-2.5 results are in
`references/guide.md`; the final LTX-2.5 recipes are in `assets/`. For another pipeline, adapt the parts of
`scripts/proxy_latent.py` named in its docstring, `ROWS` in `scripts/size.py` and `RENAMES` in
`scripts/reference.py`.

## Deliverable

`recipe.json` with one set of `nncf.compress_weights()` kwargs per component:

```json
{
  "transformer": {"mode": "int4_asym", "group_size": 128, "ratio": 1.0,
                  "scoped_configs": [{"patterns": ["__module\\.model\\.transformer_blocks\\.\\d+\\.attn1\\.to_(v|out\\.0)/"],
                                      "mode": "int8_asym", "group_size": -1}]},
  "connectors": {"mode": "int4_asym", "group_size": 128, "ratio": 1.0},
  "text_encoder": {"mode": "int4_asym", "group_size": 128, "ratio": 1.0}
}
```

- A component missing from the recipe is not compressed. Keys starting with `_` are notes.
- OpenVINO node names keep the block index (`__module.model.transformer_blocks.12.attn1.to_q/ov_ext::linear/MatMul`),
  so `patterns` can select single blocks.

## Tools

| script | what it does |
|---|---|
| `arch.py` | Inventory: the compressible weights of each component (`nodes_<component>.json`) and each family's share |
| `reference.py` | Reads a GGUF header on the HF Hub over HTTP ranges and maps each tensor's quant type to node names |
| `size.py` | Predicts recipe size and prints the per-layer-type precision table |
| `proxy_latent.py data` | Runs the BF16 pipeline once and records the teacher signals |
| `proxy_latent.py eval` | Compresses components in memory and measures errors vs the teacher: the layer-sensitivity check |
| `run_proxy.sh` | Runs `proxy_latent.py eval` on a list of recipes |
| `summarize.py` | Table of proxy results with sizes |
| `recipe_io.py` | Turns recipe.json into NNCF objects; `compress_pipeline(pipe, path)` applies it |
| `apply_recipe.py` | Writes a compressed copy of the export; unchanged parts are symlinked |
| `run_wwb.sh` | `apply_recipe.py` plus WWB scoring of one recipe: the real accuracy check |
| `wwb_ltx2_distilled.py` | WWB Text2Video evaluator with the distilled sampling recipe; also builds the reference videos |
| `wwb_compare.py` | WWB means plus the paired difference of each metric vs a reference run, with SE |
| `numa_queue.sh` | Runs queued commands one at a time, pinned to a NUMA node; start one worker per node |

## Setup

```bash
S=<this skill's directory>/video_diffusion/scripts
export PY=<python with openvino, optimum-intel, diffusers (main), torch, whowhatbench>
export EXPORT=<optimum-intel export dir>
export NNCF_SRC=<nncf checkout>/src     # if the env's NNCF has no nncf.ScopedConfig
export PYTHONPATH=$NNCF_SRC:$S
mkdir -p work/recipes logs
```

Run Python with `-P`, not `-I`: `-I` ignores PYTHONPATH and silently loads the older installed NNCF. Use `-I`
only for `reference.py`, which needs no NNCF. Queue workers inherit this environment.

## Steps

1. **Inventory.** Writes `work/nodes_*.json` and prints each family's share.

   ```bash
   $PY -P $S/arch.py $EXPORT --out-dir work
   ```

2. **Reference** (optional). Unsloth publishes LTX-2.x GGUF files on the HF Hub; only the header is downloaded.

   ```bash
   $PY -I $S/reference.py unsloth/LTX-2.5-GGUF distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf \
       --nodes work/nodes_transformer.json work/nodes_connectors.json --out work/reference_ud.json
   ```

3. **Candidates and size.** Write the plain base, the attribution recipes (one component at INT8 each), the
   cheap set and the reference translation. Check their sizes; the first recipe is the reference for the %:

   ```bash
   $PY $S/size.py work/recipes/base.json work/recipes/cheap.json ... --nodes-dir work [--no-table]
   ```

4. **Teacher data.** About 3 min per prompt, 6 GB for 6 prompts at 768×448, 49 frames.

   ```bash
   $PY -P $S/proxy_latent.py data $EXPORT --out work/proxy_data.pt
   ```

5. **Proxy rounds.** Plain `run_proxy.sh` lines run the full proxy; `--tf-only` screens transformer families
   and `--te-only` screens the text encoder and connectors. Time per candidate on a 32-core socket: full about
   25 min, `--tf-only` about 11 min, `--te-only` 2–5 min.

   ```bash
   cat >> work/queue.txt <<'EOF'
   run_proxy.sh work/results.jsonl work/recipes/base.json work/recipes/cheap.json
   run_proxy.sh work/results.jsonl --tf-only work/recipes/s_attn1_i8.json work/recipes/s_ff_g32.json
   run_proxy.sh work/results.jsonl --te-only work/recipes/te_i8.json
   EOF
   nohup $S/numa_queue.sh 0 work/queue.txt > logs/queue0.log 2>&1 &
   nohup $S/numa_queue.sh 1 work/queue.txt > logs/queue1.log 2>&1 &
   $PY $S/summarize.py work/results.jsonl --base base --exclude-prompts 3
   ```

   The order that worked:
   1. Attribution: which component limits quality.
   2. The cheap set.
   3. Family screens: each is the cheap set plus one family at INT8 or at a finer group.
   4. Combine the levers.
   5. Upgrade the component that limits free-running error.
6. **Real eval: WWB.** Make the reference videos once from the PyTorch BF16 pipeline, then queue the base, the
   chosen recipe and the one or two alternatives the proxy could not separate:

   ```bash
   $PY $S/wwb_ltx2_distilled.py --base-model <diffusers model dir or HF id> --gt-data wwb_ref/gt.csv
   export GT=wwb_ref/gt.csv      # GT_ROOT: the dir the video paths in gt.csv are relative to
   cat >> work/queue.txt <<'EOF'
   run_wwb.sh work/recipes/base.json base > logs/wwb_base.log 2>&1
   run_wwb.sh work/recipes/r2.json r2 > logs/wwb_r2.log 2>&1
   EOF
   $PY $S/wwb_compare.py --ref base --wwb-dir wwb
   ```

   - About 2.7 h per run on one socket (10 prompts, 960×544, 121 frames).
   - Each export takes 18–23 GB on disk; unchanged components are symlinked. An export with `recipe.json` is
     complete and is reused.
   - An empty recipe (`{}`) scores the uncompressed export, the upper bound.
7. **Deliver** what the method's last step lists, with:
   - one recipe per component;
   - the `size.py` table on the base and the recipe;
   - the WWB paired-difference table.

   To apply the recipe:

   ```python
   from recipe_io import compress_pipeline  # $S on sys.path

   pipe = OVLTX2Pipeline.from_pretrained(export_dir, compile=False)
   compress_pipeline(pipe, "recipe.json")
   ```

   or write a ready-to-run export: `$PY -P $S/apply_recipe.py $EXPORT recipe.json <out_dir>`.

## Reading the proxy

All metrics are relative squared errors vs BF16, `||y - y_bf16||^2 / ||y_bf16||^2`. Lower is better.

- **`tf_video`/`tf_audio`**: teacher-forced transformer output error, over 3 prompts × 8 steps. This is the
  ranking signal for layer sensitivity, and it is stable.
- **`fr_*`**: free-running final-latent error. It includes the text encoder, the connectors and error build-up
  over steps. It plateaus at the weakest component, so a flat `fr` after a transformer upgrade means another
  component is the limit, not that the upgrade did nothing.
- **Chaotic prompts**: some prompts diverge for every recipe, including all-INT8 (on LTX-2.5 the bread-kneading
  prompt lands at about 0.2 or 0.6 at random). Use `fr_geo` with `--exclude-prompts`.
- **`te`/`conn_*`**: text-encoder and connector output error. Screen them with `--te-only`, which is fast.
- **Noise**: differences under about 2× need WWB to separate them.

## Pitfalls

- **Regex scope.** Patterns use `re.match` on the full node name: `.*to_gate_logits/` also matches the
  connectors.
- **NNCF backup layers.** Without `all_layers`, NNCF puts Gather embeddings and the last MatMul of each
  component into INT8_ASYM; `size.py` accounts for them. On LTX-2.5: transformer `proj_out`, text encoder
  `layers.47.mlp.down_proj`, and the last `ff.net.2` of `audio_connector`.
- **Inference config.** The proxy runs OpenVINO in f32 with `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`, so the errors
  come from the weights only. `run_wwb.sh` uses the same config, so the two agree.
- **WWB noise.** `similarity` is the cosine similarity of ViViT embeddings (a 224×224 center crop) of the
  target and reference videos. Compare runs prompt by prompt: `wwb_compare.py` prints the mean difference and
  its SE. With 10 prompts the SE of a similarity difference is 0.01–0.03, so read a difference under 2 SE as
  noise, and check SSIM, LPIPS and tLPIPS too. On LTX-2.5 the sym recipe moved similarity by +0.011 (0.5 SE),
  while SSIM (+2.4 SE) and LPIPS (−1.9 SE) improved on 8 of 10 prompts.
- **WWB frames.** All four WWB metrics score only the first 32 of the 121 frames (ViViT's input length), so
  errors that build up later in the clip are not measured. The proxy's `fr_video` covers every frame of its
  49-frame clip.
- **Audio.** WWB scores video only. Check audio with `tf_audio`/`fr_audio`, and by listening.
- **Cost.** Spend the 2.7 h WWB runs on the base, the chosen recipe and the alternatives the proxy cannot
  separate. Keep the queue full so both sockets stay busy.
