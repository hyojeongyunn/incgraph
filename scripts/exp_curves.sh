#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
. scripts/lanes.sh
TASKS=${TASKS:-"1 2"}
STEPS=${STEPS:-all}
REPS=${REPS:-3}
NQ=${NQ:-10000}
L1=${L1:-"10,15,20,30,50,75,100,150,200,300"}
L2=${L2:-"10,15,20,30,40,50,75,100,150"}
T1_DATASETS=${T1_DATASETS:-"webvid-mm t2i-mm"}
T2_DATASETS=${T2_DATASETS:-"webvid-1M t2i-1M coco-clip"}
OUT=results/curves
list=$(mktemp)

incremental_jobs() {
  local ds rb d method f step last o
  for ds in $T1_DATASETS; do
    for d in results/snapshots/$ds/mm-clustered*-s*/*; do
      [ -d "$d" ] || continue
      method=$(basename "$d"); rb=$(basename "$(dirname "$d")")
      last=$(ls "$d" | sed -n 's/^step\([0-9]*\)\.graph$/\1/p' | sort -n | tail -1)
      for f in "$d"/step*.graph; do
        step=$(basename "$f" .graph); step=${step#step}
        if [ "$1" = last ]; then [ "$step" = "$last" ] || continue; else [ "$step" = "$last" ] && continue; fi
        o=$OUT/$ds/$rb/$method/step$step.jsonl
        [ -f "$o" ] && continue
        mkdir -p "$(dirname "$o")"
        echo "rm -f $o.part; bin/ig_curve --dataset $ds --runbook $rb --step $step --graphs $f --labels $method --nq $NQ --reps $REPS --search-L $L1 --out $o.part && mv $o.part $o"
      done
    done
  done
}

adaptive_jobs() {
  local w wl ds s idx first eps m o ov
  for w in $(for d in $T2_DATASETS; do echo results/snapshots/$d/*-s*; done); do
    [ -d "$w" ] || continue
    wl=$(basename "$w"); ds=$(basename "$(dirname "$w")"); s=${wl##*-s}
    idx=results/indexes/$ds-R32-L100-a1.2-s$s.graph
    first=$(ls -d "$w"/comember-* 2>/dev/null | head -1)
    [ -n "$first" ] || continue
    eps=$(ls "$first" | sed -n 's/^epoch\([0-9]*\)\.edges$/\1/p' | sort -n | paste -sd,)
    for m in noop $(ls "$w" | grep '^comember-'); do
      o=$OUT/$ds/$wl/$m.jsonl
      [ -f "$o" ] && continue
      mkdir -p "$(dirname "$o")"
      ov=""; [ "$m" != noop ] && ov="--overlays $w/$m"
      echo "rm -f $o.part; bin/ig_curve --dataset $ds --workload $wl --index $idx $ov --epochs $eps --label $m --reps $REPS --search-L $L2 --out $o.part && mv $o.part $o"
    done
  done
}

if [[ " $TASKS " == *" 1 "* && $STEPS != rest ]]; then incremental_jobs last >> "$list"; fi
if [[ " $TASKS " == *" 2 "* ]]; then adaptive_jobs >> "$list"; fi
if [[ " $TASKS " == *" 1 "* && $STEPS != last ]]; then incremental_jobs rest >> "$list"; fi
echo "$(wc -l < "$list") measurement jobs"
run_lanes "$list" results/logs/curves "${LANES:-0}"
rm -f "$list"
echo ALLDONE
