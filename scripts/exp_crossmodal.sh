#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4"}
SETS=${SETS:-"coco-clip coco-clip-ho"}
ROUNDS=${ROUNDS:-20}
LOG=results/logs/crossmodal
mkdir -p $LOG

run3() {
  bin/ig_crossmodal --dataset $1 $2 --policy noop > $LOG/$3-noop.log 2>&1 &
  bin/ig_crossmodal --dataset $1 $2 --policy comember --R-extra 16 --rounds $ROUNDS -p removal=score --tag score \
    > $LOG/$3-score.log 2>&1 &
  bin/ig_crossmodal --dataset $1 $2 --policy comember --R-extra 16 --rounds $ROUNDS -p removal=none --tag none \
    > $LOG/$3-none.log 2>&1 &
}

for ds in $SETS; do
  case $ds in
    coco-clip*)
      if [ ! -f data/$ds/dataset.json ]; then
        [ -d data/coco_openclip_l ] || { echo "skip $ds: data/coco_openclip_l not found"; continue; }
        bin/ig_make_coco $([ $ds = coco-clip-ho ] && echo --holdout) || exit 1
      fi
      for s in $SEEDS; do
        [ -f results/indexes/$ds-R32-L100-a1.2-s$s.graph ] || bin/ig_build --dataset $ds --seed $s > $LOG/$ds-s$s-build.log 2>&1 &
      done
      wait
      [ -f data/$ds/gt/nq2000-k10.bin ] || bin/ig_crossmodal --dataset $ds --seed 0 --rounds 0 --eval-at 0 --search-L 10 \
        --out /dev/null > $LOG/$ds-gt.log 2>&1
      for s in $SEEDS; do run3 $ds "--seed $s" $ds-s$s; done
      ;;
    t2i-mm|webvid-mm)
      [ -f data/$ds/dataset.json ] || { echo "skip $ds: data/$ds not found"; continue; }
      for s in $SEEDS; do run3 $ds "--runbook mm-clustered-s$s" $ds-mm-clustered-s$s; done
      ;;
    *) echo "unknown set $ds" >&2; exit 1 ;;
  esac
done
wait
grep -l error $LOG/*.log 2>/dev/null && echo SOME_FAILED
echo ALLDONE
