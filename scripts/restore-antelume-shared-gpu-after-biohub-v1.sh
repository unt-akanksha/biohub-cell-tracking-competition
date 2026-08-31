#!/usr/bin/env bash
set -euo pipefail

localizer_terminal=/home/ubuntu/biohub-temporal-localizer-v2/run.complete
recovery_terminal=/home/ubuntu/biohub-graph-context-recovery-v1/run.complete
for _poll in $(seq 1 1500); do
  if test -f "$recovery_terminal"; then
    break
  fi
  if \
    ! pgrep -f 'run-antelume-1gpu-temporal-localizer-v2.sh|train_synthetic_localizer.py' >/dev/null \
    && ! pgrep -f 'run-antelume-graph-context-division-recovery-v1.sh|train_graph_context_division_sweep.py' >/dev/null; then
    break
  fi
  sleep 60
done
systemctl --user unmask bigmembers.service plw7.service
sudo -n systemctl unmask rsna-plw.service
printf '%s\n' "shared GPU service masks released" \
  > /home/ubuntu/biohub-graph-context-recovery-v1/gpu-lease-restored
