#!/usr/bin/env bash
# Poll a kernel to terminal state so the session is woken promptly.
KERNEL="$1"
for i in $(seq 1 90); do
  S=$(kaggle kernels status "indarkarhana/$KERNEL" 2>&1 | head -1)
  echo "$(date -u +%H:%M) $S"
  case "$S" in *RUNNING*|*QUEUED*) sleep 240;; *) echo TERMINAL; break;; esac
done
