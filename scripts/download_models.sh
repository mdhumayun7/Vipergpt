#!/usr/bin/env bash
# [LOGIN NODE ONLY] Download pretrained weights. Compute nodes have no internet.
#
# SCOPE: Milestone 1 (codegen-only analysis) needs ONLY the program generator.
# Perception modules (GLIP / X-VLM / MiDaS) are fetched by the --full flag and
# are NOT required until Milestone 2.
#
# Usage:
#   bash scripts/download_models.sh           # codegen only  (~16 GB)
#   bash scripts/download_models.sh --full    # + perception  (~30 GB)
set -euo pipefail

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-$HOME/hpc-prog/humayun/vipergpt_store/pretrained_models}"
export HF_HOME="${HF_HOME:-$HOME/hf_cache}"
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

mkdir -p "$PRETRAINED_MODEL_PATH" "$HF_HOME"
echo "Weights  -> $PRETRAINED_MODEL_PATH"
echo "HF cache -> $HF_HOME"

# --- disk guard: refuse to start if headroom is thin ---------------------------
AVAIL_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
NEED_GB=$([[ $FULL -eq 1 ]] && echo 35 || echo 20)
if (( AVAIL_GB < NEED_GB )); then
  echo "ABORT: only ${AVAIL_GB}G free on the filesystem, need ~${NEED_GB}G." >&2
  exit 1
fi
echo "Filesystem headroom: ${AVAIL_GB}G — proceeding."

# --- 1. program generator pi (DEVIATION D1: replaces deprecated Codex) ---------
echo ">>> Qwen2.5-Coder-7B-Instruct"
hf download Qwen/Qwen2.5-Coder-7B-Instruct

# --- 2. perception modules (Milestone 2 only) ---------------------------------
if [[ $FULL -eq 1 ]]; then
  echo ">>> MiDaS (compute_depth)"
  hf download Intel/dpt-hybrid-midas || huggingface-cli download Intel/dpt-hybrid-midas

  echo ">>> GLIP-L checkpoint (find / exists)"
  mkdir -p "$PRETRAINED_MODEL_PATH/GLIP/checkpoints" "$PRETRAINED_MODEL_PATH/GLIP/configs"
  wget -c -O "$PRETRAINED_MODEL_PATH/GLIP/checkpoints/glip_large_model.pth" \
    https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
  wget -c -O "$PRETRAINED_MODEL_PATH/GLIP/configs/glip_Swin_L.yaml" \
    https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml

  echo "NOTE: X-VLM weights are distributed via Google Drive by the official repo."
  echo "      Fetch manually and place at \$PRETRAINED_MODEL_PATH/xvlm/ ."
  echo "      See docs/deviations.md — models/_xvlm_backbone.py is still a stub."
fi

# --- 3. validate by WEIGHT FILES, not config.json ------------------------------
echo
echo "=== Validation (weight files + sizes) ==="
find "$HF_HOME" "$PRETRAINED_MODEL_PATH" \
     \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pth" \) \
     -size +10M -exec ls -lh {} \; 2>/dev/null \
  | awk '{print $5, $9}' | sort -k2 || true

TOTAL=$(du -sh "$HF_HOME" "$PRETRAINED_MODEL_PATH" 2>/dev/null | awk '{s=$1; print $1"\t"$2}')
echo
echo "$TOTAL"
echo "Done. Next: HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_offline.py"