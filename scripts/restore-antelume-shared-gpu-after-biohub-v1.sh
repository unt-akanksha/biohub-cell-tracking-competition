#!/usr/bin/env bash
set -euo pipefail

localizer_terminal=/home/ubuntu/biohub-temporal-localizer-v2/run.complete
recovery_terminal=/home/ubuntu/biohub-graph-context-recovery-v1/run.complete
for _poll in $(seq 1 6000); do
  if test -f "$recovery_terminal"; then
    break
  fi
  if \
    ! pgrep -f 'run-antelume-1gpu-temporal-localizer-v2.sh|train_synthetic_localizer.py' >/dev/null \
    && ! pgrep -f 'run-antelume-graph-context-division-recovery-v1.sh|train_graph_context_division_sweep.py' >/dev/null; then
    break
  fi
  sleep 15
done
printf '%s\n' \
  "shared GPU services were not changed" \
  "localizer_complete=$(test -f "$localizer_terminal" && printf true || printf false)" \
  "recovery_complete=$(test -f "$recovery_terminal" && printf true || printf false)" \
  > /home/ubuntu/biohub-graph-context-recovery-v1/gpu-lease-restored
