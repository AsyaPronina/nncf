---
name: scoped-config-recipe
description: Choose per-layer precision and group size for NNCF data-free weight compression of a PyTorch/HF or OpenVINO model and write it as a recipe.json of nncf.compress_weights() parameters (scoped_configs + ignored_scope; one set per component for a multi-component pipeline). Has routes for LLMs and for video-diffusion pipelines such as LTX-2.x. Uses the model architecture, an optional reference quantization (e.g. an Unsloth UD GGUF), a base precision, a proxy eval vs BF16 and a real eval. Use whenever the user asks which layers need higher precision, wants a mixed INT4/INT8 or "Unsloth-style" recipe, mentions scoped_configs or per-layer sensitivity, or wants roughly INT4 size without losing accuracy.
compatibility: NNCF develop (nncf.ScopedConfig). llm route - torch, transformers, datasets, a CUDA GPU for models beyond a few B params, vLLM for the proxy data. video_diffusion route - openvino, optimum-intel, diffusers, whowhatbench; CPU only.
---

# Scoped-config recipe

Finds which layers of a model need more than the base precision and writes the answer as `recipe.json`, the
keyword arguments of `nncf.compress_weights()`.

## Usage

Inputs:

- **Model**: a PyTorch/HF checkpoint or an OpenVINO IR (e.g. an optimum-intel export).
- **Base precision**: e.g. `int4_asym` g128. Everything is measured against it.
- **Size limit** (optional): GiB or % over the base.
- **Reference quantization** (optional): e.g. an Unsloth UD GGUF of the same model.

Then:

1. Find the model's kind and format in the routes table. If no route fits, tell the user instead of forcing one.
2. Read the route's two files and follow its `ROUTE.md`. Paths inside a route are relative to the route's folder.

## Routes

| model kind | PyTorch / HF | OpenVINO IR | files to read |
|---|---|---|---|
| LLM: dense, MoE, hybrid SSM | `llm/` | no route yet | `llm/ROUTE.md`, `llm/references/guide.md` |
| video-diffusion pipeline, e.g. LTX-2.x | export it first: `optimum-cli export openvino` | `video_diffusion/` | `video_diffusion/ROUTE.md`, `video_diffusion/references/guide.md` |

Every route has the same layout: `ROUTE.md` (setup and steps), `references/guide.md` (priors per weight family
and measured results), `assets/` (measured recipes) and `scripts/`.

## Method

Every route implements these steps with its own tools and metrics:

1. **Inventory**: every compressible weight, grouped into families, with each family's share of the weights.
2. **Reference**: translate a reference quantization instead of copying it. Q8_0 / Q6_K / BF16 → INT8_ASYM
   per-channel; one level above the base type → INT8 on small families, a finer INT4 group on large ones.
3. **Candidates**: the plain base plus 2–4 recipes that each test one idea, with their sizes. Start with the
   cheap upgrades: small, always-on families to INT8_ASYM. INT8 is the upgrade target; BF16 is almost never
   worth its size.
4. **Proxy**: score every candidate against BF16, and find which component and which family limit quality.
5. **Refine** for 1–2 rounds: push the changes that paid off for their size, revert the ones that didn't.
   Proxy differences under ~2× need a real eval to separate.
6. **Real eval and deliver**: the recipe, the per-layer-type precision table (base vs recipe), one line of
   reasoning and evidence per changed layer type, the proxy score of every candidate per round, and the real
   eval result, or the eval the user should run to confirm.

## Recipe format

`recipe.json` holds `nncf.compress_weights()` kwargs: `mode`, `group_size`, `ratio`, `scoped_configs`,
`ignored_scope`, ... For a multi-component pipeline it maps each component to its own kwargs:
`{"transformer": {...}, "text_encoder": {...}}`.

- Scoped configs and the ignored scope take nncf.Scope fields: `names`, `patterns` (`re.match` on node names),
  `types`, `subgraphs`.
- The last matching scoped config wins, and a scoped config overrides `ignored_scope`.
- Node names differ between the PyTorch and OpenVINO backends, so a recipe only fits the backend it was built on.

## Adding a route

Create `<route>/` with the same layout, implement the method's six steps, and add it to the routes table: a new
row for a new model kind, or a filled-in cell for a new format of an existing kind.
