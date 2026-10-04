#!/usr/bin/env bash
# Fast experiment launcher: register -> preflight -> authorize -> execute.
# One kernel at a time; the guard refuses a second while one is active.
set -euo pipefail
RUN_ID="$1"; KERNEL="$2"; HOURS="${3:-3.0}"; HYP="${4:-Fast experiment.}"
PY=/c/Users/IndarKumar/AppData/Local/Programs/Python/Python312/python.exe
SP="C:/Users/INDARK~1/AppData/Local/Temp/claude/c--Users-IndarKumar-Documents-Comp-Biohub/5eb9f52b-4703-4663-b61f-e762a52d6059/scratchpad"
export PYTHONIOENCODING=utf-8
$PY -m biohub_tracker experiment register --run-id "$RUN_ID" --max-runtime-hours "$HOURS" \
  --split "hidden test set (production); validator disabled" \
  --config "kaggle/$KERNEL/build-manifest.json" --hypothesis "$HYP" >/dev/null
rm -f "artifacts/preflights/$RUN_ID.json"
D4_PREFLIGHT_RUN_ID="$RUN_ID" D4_PREFLIGHT_KERNEL="$KERNEL" D4_PREFLIGHT_NO_D4=1 \
  $PY scripts/build-d4-complete-preflight.py --output "artifacts/preflights/$RUN_ID.json" >/dev/null
$PY -m biohub_tracker launch authorize --run-id "$RUN_ID" --kernel-dir "kaggle/$KERNEL" \
  --kernel-ref "indarkarhana/$KERNEL" --preflight-report "artifacts/preflights/$RUN_ID.json" \
  --live > "$SP/auth_$RUN_ID.json"
AID=$($PY -c "import json;print(json.load(open(r'$SP/auth_$RUN_ID.json'))['authorization']['authorization_id'])")
NON=$($PY -c "import json;print(json.load(open(r'$SP/auth_$RUN_ID.json'))['authorization']['nonce'])")
$PY -m biohub_tracker launch execute --authorization-id "$AID" --nonce "$NON" --execute --live >/dev/null
echo "LAUNCHED $RUN_ID -> indarkarhana/$KERNEL"
kaggle kernels status "indarkarhana/$KERNEL" 2>&1 | head -1
