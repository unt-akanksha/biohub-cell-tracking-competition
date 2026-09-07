#!/usr/bin/env bash
set -euo pipefail

# This guard owns only the Biohub workloads named below.  It never sends a
# signal to an unrelated process: another GPU client merely causes the Biohub
# workload to yield until that client has released the GPU.
state_root=/home/ubuntu/biohub-gpu-yield-guard-v1
paused_file="$state_root/paused-pids"
mode_file="$state_root/mode"
log_file="$state_root/events.log"
once=false
if [[ "${1:-}" == "--once" ]]; then
  once=true
fi

mkdir -p "$state_root"
exec 9>"$state_root/guard.lock"
flock -n 9 || exit 0

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

is_owned_biohub_gpu_pid() {
  local pid="$1"
  local cwd command
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  case "$cwd" in
    /home/ubuntu/biohub-temporal-localizer-v2/*)
      [[ -r "/proc/$pid/cmdline" ]] || return 1
      command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ \
        "$command" == *"train_synthetic_localizer.py"* \
        || "$command" == *"score_real_development_probe.py"* \
      ]]
      ;;
    /home/ubuntu/biohub-graph-context-recovery-v1/*)
      [[ -r "/proc/$pid/cmdline" ]] || return 1
      command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ \
        "$command" == *"train_graph_context_division_sweep.py"* \
        || "$command" == *"score_graph_context_division_development_probe.py"* \
      ]]
      ;;
    /home/ubuntu/biohub-peak-rank-detector-v1/*)
      [[ -r "/proc/$pid/cmdline" ]] || return 1
      command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ "$command" == *"train_synthetic_real_detector.py"* ]]
      ;;
    /home/ubuntu/biohub-nucverse3d-compatibility-v1/*)
      [[ -r "/proc/$pid/cmdline" ]] || return 1
      command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ "$command" == *"/home/ubuntu/biohub-nucverse3d-compatibility-v1/input/screen_pretrained.py"* ]]
      ;;
    /home/ubuntu/biohub)
      [[ -r "/proc/$pid/cmdline" ]] || return 1
      command=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ \
        "$command" == *"/home/ubuntu/biohub-graph-context-recovery-v1/input/train_graph_context_division_sweep.py"* \
        || "$command" == *"score_graph_context_division_development_probe.py"* \
        || "$command" == *"/home/ubuntu/biohub-peak-rank-detector-v1/input/train_synthetic_real_detector.py"* \
      ]]
      ;;
    *)
      return 1
      ;;
  esac
}

read_gpu_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | grep -E '^[0-9]+$' \
    || true
}

record_paused_pid() {
  local pid="$1"
  local temporary="$state_root/paused-pids.tmp"
  { test -f "$paused_file" && cat "$paused_file" || true; printf '%s\n' "$pid"; } \
    | sort -nu > "$temporary"
  mv "$temporary" "$paused_file"
}

set_mode() {
  local mode="$1"
  local previous=""
  if test -f "$mode_file"; then
    previous=$(cat "$mode_file")
  fi
  if [[ "$previous" != "$mode" ]]; then
    printf '%s mode=%s\n' "$(timestamp)" "$mode" >> "$log_file"
    printf '%s\n' "$mode" > "$mode_file"
  fi
}

while true; do
  mapfile -t gpu_pids < <(read_gpu_pids)
  owned_pids=()
  unrelated_gpu_client=false
  for pid in "${gpu_pids[@]}"; do
    if is_owned_biohub_gpu_pid "$pid"; then
      owned_pids+=("$pid")
    else
      unrelated_gpu_client=true
    fi
  done

  if "$unrelated_gpu_client"; then
    for pid in "${owned_pids[@]}"; do
      if kill -s STOP "$pid" 2>/dev/null; then
        record_paused_pid "$pid"
      fi
    done
    set_mode "biohub-paused-for-unrelated-gpu-client"
  else
    if test -s "$paused_file"; then
      while IFS= read -r pid; do
        if is_owned_biohub_gpu_pid "$pid"; then
          kill -s CONT "$pid" 2>/dev/null || true
        fi
      done < "$paused_file"
      : > "$paused_file"
    fi
    set_mode "biohub-allowed"
  fi

  if "$once"; then
    break
  fi
  sleep 2
done
