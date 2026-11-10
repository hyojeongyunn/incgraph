#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4"}
JOBS=${JOBS:-8}
BUDGET=${BUDGET:-300000}
DATASETS=${DATASETS:-"t2i-1M webvid-1M coco-clip"}
WORKLOADS=${WORKLOADS:-"abrupt32 gradual32-long"}
LOG=results/logs/adaptive
mkdir -p "$LOG"

echo "[1/3] datasets and workloads"
avail=""
for ds in $DATASETS; do
  if [ ! -f "data/$ds/dataset.json" ] && [ "$ds" = coco-clip ] && [ -d data/coco_openclip_l ]; then
    bin/ig_make_coco || exit 1
  fi
  if [ -f "data/$ds/dataset.json" ]; then avail="$avail $ds"; else echo "skip $ds: data/$ds not found"; fi
done
DATASETS=$avail
jobs=()
for ds in $DATASETS; do for s in $SEEDS; do
  [ -f "data/$ds/workloads/abrupt32-s$s/meta.json" ] || jobs+=("bin/ig_make_drift --dataset $ds --name abrupt32 --drift abrupt --epochs 12 --period 3 --seed $s > $LOG/$ds-abrupt32-s$s-make.log 2>&1")
  [ -f "data/$ds/workloads/gradual32-long-s$s/meta.json" ] || jobs+=("bin/ig_make_drift --dataset $ds --name gradual32-long --drift gradual --epochs 30 --seed $s > $LOG/$ds-gradual32-long-s$s-make.log 2>&1")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} bash -c '{}'

echo "[2/3] static indexes"
jobs=()
for ds in $DATASETS; do for s in $SEEDS; do
  [ -f "results/indexes/$ds-R32-L100-a1.2-s$s.graph" ] || jobs+=("bin/ig_build --dataset $ds --R 32 --seed $s > $LOG/$ds-build-s$s.log 2>&1")
done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'

echo "[3/3] methods"
jobs=()
for ds in $DATASETS; do for wl in $WORKLOADS; do for s in $SEEDS; do
  w="$wl-s$s"
  every=1; [ "$wl" = gradual32-long ] && every=3
  base="bin/ig_drift --dataset $ds --workload $w --seed $s --snap-every $every --search-threads 2"
  for v in "noop|--policy noop" \
           "score|--policy comember --R-extra 16 -p removal=score" \
           "none|--policy comember --R-extra 16 -p removal=none" \
           "score-B|--policy comember --R-extra 16 -p removal=score -p budget=$BUDGET" \
           "none-B|--policy comember --R-extra 16 -p removal=none -p budget=$BUDGET"; do
    name=${v%%|*}; args=${v#*|}
    tag=""; [ "$name" != noop ] && tag="--tag $name"
    lbl=$name; [ "$name" != noop ] && lbl="comember-$name"
    out=results/$ds/$w/$lbl.jsonl
    [ -f "$out" ] && grep -q '"type":"summary"' "$out" && continue
    jobs+=("$base $args $tag > $LOG/$ds-$w-$name.log 2>&1 || echo FAIL $ds $w $name")
  done
done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'
echo ALLDONE
