#!/usr/bin/env bash
# Run queued commands one at a time, pinned to one NUMA node. Start one worker per node on the same queue:
#   numa_queue.sh <numa_node|all> <queue_file>
# A queue line is a shell command, run from the worker's directory with this folder on PATH, e.g.
#   run_proxy.sh work/results.jsonl --tf-only work/recipes/s_attn1_i8.json
#   run_wwb.sh work/recipes/base.json base > logs/wwb_base.log 2>&1
# Lines can be appended while the workers run; a worker exits when the queue is empty.
set -uo pipefail
node=$1; queue=$2
here=$(cd "$(dirname "$0")" && pwd)
export PATH=$here:$PATH
bind=(); [ "$node" != all ] && bind=(numactl --cpunodebind="$node" --membind="$node")
pop='n=$(grep -n -m 1 "[^[:space:]]" "$0" | cut -d: -f1); [ -n "$n" ] && sed -n "${n}p" "$0" && sed -i "1,${n}d" "$0"'
while true; do
    job=$(flock "$queue.lock" bash -c "$pop" "$queue" 2>/dev/null)
    [ -z "$job" ] && break
    echo "[$(date +%H:%M:%S)] node $node: $job"
    if "${bind[@]}" bash -c "$job"; then
        echo "[$(date +%H:%M:%S)] done: $job"
    else
        echo "[$(date +%H:%M:%S)] FAILED: $job"
    fi
done
echo "[$(date +%H:%M:%S)] node $node: queue empty"
