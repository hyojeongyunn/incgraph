run_lanes() {
  local jobs=$1 log=$2 cores=${3:-0}
  local queue lock core
  queue=$(mktemp)
  lock=$queue.lock
  cp "$jobs" "$queue"
  mkdir -p "$log"
  for core in $cores; do
    (
      while :; do
        job=$( { flock 9; j=$(head -n 1 "$queue"); [ -n "$j" ] && sed -i 1d "$queue"; printf '%s' "$j"; } 9>"$lock" )
        [ -n "$job" ] || break
        echo "[$(date '+%F %T')] start: $job" >> "$log/lane$core.log"
        if taskset -c "$core" bash -c "$job" >> "$log/lane$core.log" 2>&1; then
          echo "[$(date '+%F %T')] done" >> "$log/lane$core.log"
        else
          echo "[$(date '+%F %T')] FAIL: $job" >> "$log/lane$core.log"
        fi
      done
    ) &
  done
  wait
  rm -f "$queue" "$lock"
}
