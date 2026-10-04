#!/usr/bin/env bash
# Sequential experiment runner. Waits for the GPU to free, launches the next
# queued kernel, waits for it to finish, repeats. Kaggle allows one active
# kernel, so this is strictly serial by necessity.
#
# Each entry is  run_id|kernel_slug|declared_hours|hypothesis
set -uo pipefail
QUEUE_FILE="$1"
while IFS='|' read -r RUN_ID KERNEL HOURS HYP; do
  [ -z "${RUN_ID:-}" ] && continue
  case "$RUN_ID" in \#*) continue;; esac
  # wait for any active kernel to clear
  for _ in $(seq 1 240); do
    ACTIVE=$(kaggle kernels status "indarkarhana/$KERNEL" 2>&1 | head -1)
    case "$ACTIVE" in *RUNNING*|*QUEUED*) sleep 180;; *) break;; esac
  done
  echo "$(date -u +%H:%M) launching $RUN_ID"
  bash scripts/fast-launch-v1.sh "$RUN_ID" "$KERNEL" "$HOURS" "$HYP" 2>&1 | tail -1
  for _ in $(seq 1 240); do
    S=$(kaggle kernels status "indarkarhana/$KERNEL" 2>&1 | head -1)
    echo "$(date -u +%H:%M) $KERNEL $S"
    case "$S" in *RUNNING*|*QUEUED*) sleep 240;; *) break;; esac
  done
  echo "$(date -u +%H:%M) $RUN_ID terminal"
done < "$QUEUE_FILE"
echo "QUEUE COMPLETE"
