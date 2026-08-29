#!/usr/bin/env bash
set -euo pipefail

workspace=/home/ubuntu/biohub
data_root=/home/ubuntu/biohub-data
kaggle_cli=/home/ubuntu/venv/bin/kaggle
log_root=/home/ubuntu/biohub-logs

if [[ "$(readlink -f "$workspace")" != /home/ubuntu/biohub ]]; then
  echo "Refusing unexpected Biohub workspace: $workspace" >&2
  exit 2
fi
if [[ "$(readlink -f "$data_root")" != /home/ubuntu/biohub-data ]]; then
  echo "Refusing unexpected Biohub data root: $data_root" >&2
  exit 2
fi
if [[ ! -f "$workspace/research/temporal_contrastive/train_dual_fold_division_localization.py" ]]; then
  echo "The isolated Biohub source archive is incomplete" >&2
  exit 2
fi
if [[ ! -x "$kaggle_cli" ]]; then
  echo "Kaggle CLI is unavailable at $kaggle_cli" >&2
  exit 2
fi

mkdir -p "$data_root" "$log_root"

download_dataset() {
  local ref=$1
  local slug=${ref#*/}
  local destination="$data_root/$slug"
  local complete_marker="$destination/.download-complete"
  if [[ -f "$complete_marker" ]]; then
    echo "Already complete: $ref"
    return
  fi
  if [[ -e "$destination" ]]; then
    echo "Refusing partial or unverified destination: $destination" >&2
    exit 3
  fi
  mkdir "$destination"
  "$kaggle_cli" datasets download -d "$ref" -p "$destination" --unzip
  find "$destination" -type f ! -name '.download-complete' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "$destination/SHA256SUMS"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$complete_marker"
  echo "Downloaded and inventoried: $ref"
}

nvidia-smi --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader > "$log_root/bootstrap-gpu.txt"

download_dataset indarkarhana/biohub-zebrahub-contextual-shards-v1
download_dataset indarkarhana/biohub-temporal-multiscale-contextual-runtime-v4
download_dataset indarkarhana/biohub-division-localization-shards-v1
download_dataset indarkarhana/biohub-division-localization-runtime-v1

du -sh "$workspace" "$data_root" > "$log_root/bootstrap-disk.txt"
find "$data_root" -mindepth 2 -maxdepth 2 -name '.download-complete' -print \
  | sort > "$log_root/bootstrap-complete-datasets.txt"
echo "Antelume Biohub data bootstrap complete"
