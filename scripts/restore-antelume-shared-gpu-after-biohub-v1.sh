#!/usr/bin/env bash
set -euo pipefail

localizer_terminal=/home/ubuntu/biohub-temporal-localizer-v2/run.complete
recovery_terminal=/home/ubuntu/biohub-graph-context-recovery-v1/run.complete
mask_record=/home/ubuntu/biohub-graph-context-recovery-v1/runtime-masked-rsna-services.txt
printf '%s\n' rsna-plw.service rsna-dump.service >"$mask_record"

for _poll in $(seq 1 6000); do
  if test -f "$recovery_terminal"; then
    break
  fi
  if \
    ! pgrep -f 'run-antelume-1gpu-temporal-localizer-v2.sh|train_synthetic_localizer.py' >/dev/null \
    && ! pgrep -f 'run-antelume-graph-context-division-recovery-v1.sh|train_graph_context_division_sweep.py' >/dev/null; then
    break
  fi
  mapfile -t active_rsna_units < <(
    systemctl --no-legend --plain --type=service --state=running \
      | awk '$1 ~ /^rsna-[A-Za-z0-9_.@-]+\.service$/ {print $1}'
  )
  for unit in "${active_rsna_units[@]}"; do
    if [[ ! "$unit" =~ ^rsna-[A-Za-z0-9_.@-]+\.service$ ]]; then
      continue
    fi
    sudo -n systemctl stop "$unit" || true
    sudo -n systemctl mask --runtime --force "$unit" || true
    if ! grep -Fxq "$unit" "$mask_record"; then
      printf '%s\n' "$unit" >>"$mask_record"
    fi
  done
  sleep 15
done
systemctl --user unmask bigmembers.service plw7.service
while IFS= read -r unit; do
  if [[ "$unit" =~ ^rsna-[A-Za-z0-9_.@-]+\.service$ ]]; then
    sudo -n systemctl unmask "$unit" || true
  fi
done <"$mask_record"
printf '%s\n' \
  "shared GPU service masks released" \
  "localizer_complete=$(test -f "$localizer_terminal" && printf true || printf false)" \
  "recovery_complete=$(test -f "$recovery_terminal" && printf true || printf false)" \
  > /home/ubuntu/biohub-graph-context-recovery-v1/gpu-lease-restored
