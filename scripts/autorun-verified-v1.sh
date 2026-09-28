#!/usr/bin/env bash
# Sequential runner that FAILS LOUDLY on a refused launch and auto-submits
# production arms as they land.
#
# The earlier runner treated a refused registration as "terminal" and moved to
# the next entry, so a hypothesis over the 1000-character limit silently cost
# two days of GPU. Here a launch must print LAUNCHED or the queue stops.
#
# Entries are  run_id|kernel_slug|declared_hours|submit(0|1)|hypothesis
set -uo pipefail
PYBIN="${PYBIN:-python}"
QUEUE_FILE="$1"
WAIT_FIRST="${2:-}"
WAIT_FIRST_DESC="${3:-Auto-submitted on completion.}"

wait_terminal () {
  for _ in $(seq 1 320); do
    S=$(kaggle kernels status "indarkarhana/$1" 2>&1 | head -1)
    case "$S" in
      *RUNNING*|*QUEUED*) sleep 180;;
      *COMPLETE*) echo "$(date -u +%H:%M) $1 COMPLETE"; return 0;;
      *) echo "$(date -u +%H:%M) $1 terminal: $S"; return 2;;
    esac
  done
  echo "$(date -u +%H:%M) WARN $1 still active after wait budget"; return 1
}

submit () {
  echo "$(date -u +%H:%M) submitting $1"
  "${PYBIN:-python}" scripts/fast-submit-v1.py "$1" "$2" 2>&1 | tail -2
}

if [ -n "$WAIT_FIRST" ]; then
  echo "$(date -u +%H:%M) waiting on in-flight $WAIT_FIRST"
  if wait_terminal "$WAIT_FIRST"; then submit "$WAIT_FIRST" "$WAIT_FIRST_DESC"; fi
fi

while IFS='|' read -r RUN_ID KERNEL HOURS DOSUB HYP; do
  [ -z "${RUN_ID:-}" ] && continue
  case "$RUN_ID" in \#*) continue;; esac
  if [ "${#HYP}" -ge 1000 ]; then
    echo "$(date -u +%H:%M) ABORT: hypothesis for $RUN_ID is ${#HYP} chars, limit 1000"; exit 1
  fi
  echo "$(date -u +%H:%M) launching $RUN_ID"
  OUT=$(bash scripts/fast-launch-v1.sh "$RUN_ID" "$KERNEL" "$HOURS" "$HYP" 2>&1)
  echo "$OUT" | tail -3
  case "$OUT" in
    *LAUNCHED*) ;;
    *) echo "$(date -u +%H:%M) ABORT: $RUN_ID did not launch; queue stopped"; exit 1;;
  esac
  if wait_terminal "$KERNEL" && [ "$DOSUB" = "1" ]; then
    submit "$KERNEL" "$HYP"
  fi
done < "$QUEUE_FILE"
echo "QUEUE COMPLETE"
