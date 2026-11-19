# Precision guide: video diffusion

## Priors per weight family (LTX-2.5 distilled, INT4 g128 base)

`tf_video` ratios are vs the plain base. Every family screen includes the cheap set (first row), which alone
gives 0.85×.

| family (share of pipeline weights) | choice | evidence |
|---|---|---|
| `time_embed`, `prompt_adaln`, `av_cross_attn_*`, every `to_gate_logits`, `proj_in`/`proj_out` (1.3%) | INT8_ASYM | Always on. tf_video 0.85×, tf_audio 0.67× for +0.2 GiB. Unsloth keeps them BF16 |
| video self-attn `attn1.to_v`, `attn1.to_out.0` (4.7%) | INT8_ASYM | attn1 is the top video error source: all INT8 gives 0.58×; v/out INT8 + q/k g32 gives 0.60× for less size |
| video self-attn `attn1.to_q/k` (4.7%) | INT4 g32 | as above |
| video FF `ff.net.0.proj`, `ff.net.2` (18.9%) | INT4 g32 | INT8 gives 0.53× for +3.1 GiB; g32 gives 0.72× for +0.35 GiB. INT8 in blocks 16–35 cut tf further but gave no WWB gain |
| video text cross-attn `attn2`, audio↔video attn, audio branch (26%) | base | INT8 gave 0.85–0.86×, the same as the cheap set alone. Unsloth puts attn2/a2v/v2a at Q2–Q3, but INT3 on them hurt in combination |
| connectors (9.3%) | base, gate logits INT8 | INT8 changes nothing: their output error comes from the text encoder |
| text encoder (35%) | base; INT8 if the budget allows (+4.9 GiB) | Sets the free-running floor once the transformer is upgraded (fr_geo 0.064 → 0.040). Partial upgrades (g32, g64, attn, mlp.down, edge layers) recover 20% at most. WWB: clear gain on the sym base, noise on asym |
| edge blocks 0, 1, 46, 47 | base | INT8 gives 0.81× for +0.7 GiB |
| INT4_ASYM vs INT4_SYM | asym | Sym has 1.1–1.3× higher proxy error. On WWB the asym recipe is ahead of the sym recipe on all four metrics (+0.03 similarity/SSIM, −0.04 LPIPS) but within 1.3 SE, so the choice rests on the proxy. Asym costs +0.2 GiB |

Check the shares in `arch.py`'s output before copying a prior to another pipeline.

The final recipes are `assets/recipe_ltx25_int4_asym.json` and `assets/recipe_ltx25_int4_sym.json`. The sym
one is the asym one with every INT4 entry switched to INT4_SYM; upgrades stay INT8_ASYM.

## Measured on LTX-2.5 distilled

WWB: 10 prompts, 960×544, 121 frames of which the first 32 are scored, vs PyTorch BF16 videos, OpenVINO f32
inference. Proxy: `fr_geo` excludes the chaotic prompt.

| recipe | GiB | tf_video | fr_geo | WWB similarity | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| OV BF16 | 63.46 | 0 | 0 | 0.972 | 0.881 | 0.096 |
| INT8_ASYM everywhere | 31.76 | 1.8e-3 | 0.024 | | | |
| INT4_ASYM g128 | 16.99 | 4.89e-2 | 0.085 | 0.831 | 0.662 | 0.334 |
| Unsloth UD-Q4_K_XL translated | 19.92 | 1.44e-2 | 0.062 | | | |
| **asym recipe** | 18.35 | 2.23e-2 | 0.064 | 0.891 | 0.719 | 0.269 |
| asym recipe + FF INT8 blocks 16–35 | 19.41 | 1.39e-2 | 0.061 | 0.899 | 0.719 | 0.264 |
| asym recipe + text encoder INT8 | 23.21 | 2.23e-2 | 0.040 | 0.896 | 0.724 | 0.252 |
| INT4_SYM g128 | 16.87 | 5.44e-2 | 0.120 | 0.852 | 0.653 | 0.354 |
| **sym recipe** | 18.15 | 2.65e-2 | 0.093 | 0.863 | 0.688 | 0.306 |
| sym recipe + text encoder INT8 | 23.05 | 2.65e-2 | 0.064 | 0.877 | 0.727 | 0.255 |

Paired results vs the matching INT4 base:

- **Asym recipe**: similarity +0.061 (SE 0.027), SSIM +0.058 (0.021), LPIPS −0.065 (0.033).
- **Sym recipe**: SSIM +0.035 (0.015) and LPIPS −0.048 (0.025). Similarity +0.011 is within noise.

Asym recipe vs sym recipe: similarity +0.029 (0.024), SSIM +0.032 (0.025), LPIPS −0.037 (0.030), tLPIPS −0.014
(0.018). All four point to asym, on 6 of 10 prompts each, but every difference is within noise.
