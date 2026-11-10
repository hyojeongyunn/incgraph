#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
SEEDS=${SEEDS:-"0 1 2 3 4"}
JOBS=${JOBS:-8}
SNAP=${SNAP:-"8,-1"}
DATASETS=${DATASETS:-"t2i-mm webvid-mm"}
RUNBOOKS=${RUNBOOKS:-"mm-clustered mm-clustered-del"}
METHODS=${METHODS:-"noop oodbridge"}
LOG=results/logs/incremental
mkdir -p "$LOG"

echo "[1/3] datasets and runbooks"
avail=""
for ds in $DATASETS; do
  src=${ds%-mm}-1M
  if [ ! -f "data/$ds/dataset.json" ] && [ -f "data/$src/dataset.json" ]; then
    bin/ig_make_mm --src "$src" --n-image 800000 --n-text 400000 || exit 1
  fi
  if [ -f "data/$ds/dataset.json" ]; then avail="$avail $ds"; else echo "skip $ds: data/$src not found (data/download.sh)"; fi
done
DATASETS=$avail
for ds in $DATASETS; do
  for s in $SEEDS; do
    for rb in $RUNBOOKS; do
      df=0; [ "$rb" = mm-clustered-del ] && df=0.5
      [ -f "data/$ds/runbooks/$rb-s$s/steps.json" ] && continue
      bin/ig_make_runbook --dataset "$ds" --init-group image --stream-group text --nclusters 16 \
        --init-nclusters 16 --delete-frac $df --name "$rb" --seed "$s" || exit 1
    done
  done
done

echo "[2/3] initial indexes and ground truth"
jobs=()
for ds in $DATASETS; do for rb in $RUNBOOKS; do for s in $SEEDS; do
  jobs+=("bin/ig_runbook --dataset $ds --runbook $rb-s$s --prepare --search-threads 4 > $LOG/$ds-$rb-s$s-prepare.log 2>&1 || echo FAIL prepare $ds $rb $s")
done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'

echo "[3/3] methods"
jobs=()
for ds in $DATASETS; do for rb in $RUNBOOKS; do for s in $SEEDS; do for m in $METHODS; do
  r="$rb-s$s"
  case $m in
    noop)      cmd="bin/ig_runbook --dataset $ds --runbook $r --policy noop --snap $SNAP" ;;
    oodbridge) cmd="bin/ig_runbook --dataset $ds --runbook $r --policy oodbridge --R-extra 1 --snap $SNAP" ;;
    *) echo "unknown method $m" >&2; exit 1 ;;
  esac
  out=results/$ds/$r/$m.jsonl
  [ -f "$out" ] && grep -q '"type":"summary"' "$out" && continue
  jobs+=("$cmd --search-threads 2 > $LOG/$ds-$r-$m.log 2>&1 || echo FAIL $ds $r $m")
done; done; done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'
echo ALLDONE
