#!/usr/bin/env bash
# Real eval: compress the export with a recipe and score it with WWB against the BF16 reference videos:
#   run_wwb.sh <recipe.json> <name>
# Writes the compressed export to $OUT/exports/<name> (reused if complete) and the metrics to $OUT/wwb/<name>.
# An empty recipe ({}) scores the uncompressed export. Pin it to a NUMA node with numa_queue.sh (or numactl).
# Env: EXPORT      optimum-intel export dir (required)
#      GT          gt.csv made by `wwb_ltx2_distilled.py --base-model ...` (required)
#      GT_ROOT     dir the video paths inside gt.csv are relative to (default: parent of gt.csv's dir)
#      PY, NNCF_SRC  as in run_proxy.sh;  OUT  output root (default: .)
#      OV_CONFIG   JSON string or file (default: f32 inference, no dynamic quantization)
#      WWB_SCRIPT  default: wwb_ltx2_distilled.py next to this script
set -euo pipefail
recipe=$(realpath "$1"); name=$2
here=$(cd "$(dirname "$0")" && pwd)
: "${EXPORT:?set EXPORT to the optimum-intel export dir}"
: "${GT:?set GT to the WWB gt.csv}"
py=${PY:-python}; out=$(realpath "${OUT:-.}"); gt=$(realpath "$GT")
gt_root=${GT_ROOT:-$(dirname "$(dirname "$gt")")}
f32='{"INFERENCE_PRECISION_HINT": "f32", "KV_CACHE_PRECISION": "f32", "DYNAMIC_QUANTIZATION_GROUP_SIZE": 0}'
ov_config=${OV_CONFIG:-$f32}
wwb=$(realpath "${WWB_SCRIPT:-$here/wwb_ltx2_distilled.py}")
export_dir=$out/exports/$name
if [ ! -f "$export_dir/recipe.json" ]; then  # apply_recipe.py writes recipe.json last
    rm -rf "$export_dir"
    PYTHONPATH=${NNCF_SRC:+$NNCF_SRC:}$here "$py" -P "$here/apply_recipe.py" "$EXPORT" "$recipe" "$export_dir"
fi
cd "$gt_root"
"$py" "$wwb" --target-model "$export_dir" --gt-data "$gt" --output "$out/wwb/$name" --ov-config "$ov_config"
