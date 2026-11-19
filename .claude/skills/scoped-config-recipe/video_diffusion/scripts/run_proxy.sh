#!/usr/bin/env bash
# Score recipes with `proxy_latent.py eval`; the teacher data is loaded once per call:
#   run_proxy.sh <results.jsonl> [--tf-only | --te-only] <recipe.json>...
# A candidate's label is its file name. Pin it to a NUMA node with numa_queue.sh (or numactl).
# Env: EXPORT    optimum-intel export dir (required)
#      PY        python with openvino, optimum-intel, diffusers, torch (default: python)
#      NNCF_SRC  NNCF develop src dir, if the env's NNCF has no nncf.ScopedConfig
#      DATA      teacher data from `proxy_latent.py data` (default: work/proxy_data.pt)
set -euo pipefail
results=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
: "${EXPORT:?set EXPORT to the optimum-intel export dir}"
args=()
for a in "$@"; do
    case $a in
        *.json) args+=(--candidate "$(basename "${a%.json}")=$a") ;;
        *) args+=("$a") ;;
    esac
done
export PYTHONPATH=${NNCF_SRC:+$NNCF_SRC:}$here${PYTHONPATH:+:$PYTHONPATH}
# -P, not -I: -I ignores PYTHONPATH and silently picks up the installed (older) NNCF.
exec "${PY:-python}" -P "$here/proxy_latent.py" eval "$EXPORT" --data "${DATA:-work/proxy_data.pt}" \
    --results "$results" "${args[@]}"
