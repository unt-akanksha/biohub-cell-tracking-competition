#!/usr/bin/env bash
# Run each queued kernel to completion and submit it, serially.
# Entries: run_id|kernel_slug|hours|message
set -uo pipefail
PY=/c/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe
while IFS='|' read -r RUN_ID KERNEL HOURS MSG; do
  [ -z "${RUN_ID:-}" ] && continue
  case "$RUN_ID" in \#*) continue;; esac
  # wait for the GPU to be free of ANY of our kernels
  for _ in $(seq 1 300); do
    BUSY=$(kaggle kernels list -m --page-size 20 2>/dev/null | awk 'NR>2{print $1}' | while read -r k; do
             st=$(kaggle kernels status "$k" 2>&1 | head -1); case "$st" in *RUNNING*|*QUEUED*) echo busy;; esac; done | head -1)
    [ -z "$BUSY" ] && break
    sleep 180
  done
  echo "$(date -u +%H:%M) launching $RUN_ID"
  bash scripts/fast-launch-v1.sh "$RUN_ID" "$KERNEL" "$HOURS" "$MSG" 2>&1 | tail -1
  for _ in $(seq 1 300); do
    S=$(kaggle kernels status "indarkarhana/$KERNEL" 2>&1 | head -1)
    case "$S" in *RUNNING*|*QUEUED*) sleep 240;; *) echo "$(date -u +%H:%M) $KERNEL $S"; break;; esac
  done
  case "$S" in *COMPLETE*) PYTHONIOENCODING=utf-8 $PY scripts/fast-submit-v1.py "$KERNEL" "$MSG" 2>&1 | tail -1;;
                        *) echo "$(date -u +%H:%M) $KERNEL not COMPLETE, skipping submit";; esac
done < "$1"
echo "QUEUE COMPLETE"
