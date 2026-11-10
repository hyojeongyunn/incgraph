#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
. scripts/lanes.sh
OUT=results/costs
LOG=results/logs/costs
mkdir -p "$OUT" "$LOG"
list=$(mktemp)
T1_DATASETS=${T1_DATASETS:-"t2i-mm webvid-mm"}
T2_DATASETS=${T2_DATASETS:-"t2i-1M webvid-1M coco-clip"}
for ds in $T1_DATASETS; do
  r=mm-clustered-s0
  [ -f "data/$ds/runbooks/$r/steps.json" ] || { echo "skip $ds: runbook $r not found"; continue; }
  echo "bin/ig_runbook --dataset $ds --runbook $r --policy noop --search-at -1 --search-threads 1 --out $OUT/$ds-$r-noop.jsonl" >> "$list"
  echo "bin/ig_runbook --dataset $ds --runbook $r --policy oodbridge --R-extra 1 --search-at -1 --search-threads 1 --out $OUT/$ds-$r-oodbridge.jsonl" >> "$list"
done
for ds in $T2_DATASETS; do
  w=abrupt32-s0
  [ -f "data/$ds/workloads/$w/meta.json" ] || { echo "skip $ds: workload $w not found"; continue; }
  for rm in score none; do
    echo "bin/ig_drift --dataset $ds --workload $w --seed 0 --policy comember --R-extra 16 -p removal=$rm --snap-every 0 --search-L 50 --search-threads 1 --out $OUT/$ds-$w-comember-$rm.jsonl" >> "$list"
  done
  echo "bin/ig_build --dataset $ds --R 32 --seed 0 --out $OUT/$ds-build-s0.graph > $OUT/$ds-build-s0.log 2>&1" >> "$list"
done
run_lanes "$list" "$LOG" "${LANES:-0}"
rm -f "$list" "$OUT"/*.graph
echo ALLDONE
