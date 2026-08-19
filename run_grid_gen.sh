#!/bin/bash
#================================================================
#  GRID GENERATION — one model, all four conditions, both datasets.
#
#      sbatch run_grid_gen.sh qwen7b
#      sbatch run_grid_gen.sh deepseek
#      sbatch run_grid_gen.sh yicoder
#      sbatch run_grid_gen.sh opencoder
#
#  Resumable: a cell whose run directory already exists is skipped, so a job
#  that is cut short can simply be resubmitted.
#
#  Ordered by value. Conditions run C1, C4, C2, C3 and RefCOCO before RefCOCO+,
#  so that if the allocation expires the headline comparison (C1 against C4) is
#  already complete for that model.
#================================================================
#SBATCH --job-name=grid_gen
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%A_%j.out
#SBATCH --error=outputs/slurm/%x_%A_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00

set -uo pipefail

KEY="${1:-}"
if [[ -z "$KEY" ]]; then
  echo "usage: sbatch run_grid_gen.sh <qwen7b|qwen1p5b|qwen32b|deepseek|yicoder|opencoder>"
  exit 2
fi

# ---------------------------------------------------------------- registry
# Ungated models only: no HuggingFace licence acceptance, no token required.
case "$KEY" in
  qwen7b)    MODEL=Qwen/Qwen2.5-Coder-7B-Instruct              ; BATCH=8  ;;
  qwen1p5b)  MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct            ; BATCH=16 ;;
  qwen32b)   MODEL=Qwen/Qwen2.5-Coder-32B-Instruct             ; BATCH=4  ;;
  deepseek)  MODEL=deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct ; BATCH=6  ;;
  yicoder)   MODEL=01-ai/Yi-Coder-9B-Chat                      ; BATCH=8  ;;
  opencoder) MODEL=infly/OpenCoder-8B-Instruct                 ; BATCH=8  ;;
  *) echo "unknown model key: $KEY"; exit 2 ;;
esac

module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1

cd /home/mazaveri/hpc-prog/humayun/vipergpt || exit 1
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p outputs/slurm outputs/runs

echo "============================================================"
echo "  GRID GENERATION"
echo "============================================================"
echo "node   : $SLURMD_NODENAME"
echo "key    : $KEY"
echo "model  : $MODEL"
echo "batch  : $BATCH"
echo "started: $(date)"
echo "============================================================"

python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU visible"; exit 1; }

# Fail early and loudly if the weights are not cached: the compute node is offline.
python - <<PY || { echo "[FATAL] weights not cached — run on the LOGIN node: hf download $MODEL"; exit 1; }
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
try:
    p = Path(snapshot_download("$MODEL", local_files_only=True))
except Exception as e:
    print("not cached:", type(e).__name__); sys.exit(1)
sh = [f for f in p.rglob("*") if f.suffix in {".safetensors", ".bin"} and f.stat().st_size > 1_000_000]
if not sh:
    print("no weight shards found"); sys.exit(1)
print(f"cached: {len(sh)} shards, {sum(f.stat().st_size for f in sh)/1e9:.1f} GB")
PY

# Short tag used in the run-directory name, matching codegen_analysis.py's scheme.
SHORT=$(python -c "print('$MODEL'.split('/')[-1].replace('Qwen2.5-Coder-','').replace('-Instruct',''))")

DONE=0; SKIP=0; FAIL=0
# Value ordering: headline comparison first, then the intermediate conditions.
for DS in refcoco refcoco+; do
  for PROMPT in api.prompt api_depth_cond.prompt api_grounding.prompt api_depth.prompt; do
    TAG=$(python -c "
from pathlib import Path
s = Path('prompts/$PROMPT').stem.replace('api_','').replace('api','base')
print(s)")
    EXPECT="outputs/runs/*m1_${DS}_testA_${SHORT}_${TAG}_greedy"
    if compgen -G "$EXPECT" > /dev/null; then
      echo ""
      echo ">>> SKIP  $DS | $PROMPT  (already generated)"
      SKIP=$((SKIP+1))
      continue
    fi
    echo ""
    echo ">>> GEN   $DS | $PROMPT | $SHORT | greedy | $(date '+%H:%M:%S')"
    python -m vipergpt_repro.eval.codegen_analysis \
        --version "$DS" --split testA --max-samples 500 \
        --model "$MODEL" --prompt "prompts/$PROMPT" \
        --batch-size "$BATCH" --temperature 0.0
    if [[ $? -eq 0 ]]; then DONE=$((DONE+1)); else FAIL=$((FAIL+1)); fi
  done
done

echo ""
echo "============================================================"
echo "  generated $DONE   skipped $SKIP   failed $FAIL"
echo "  finished: $(date)"
echo "============================================================"
ls -1dt outputs/runs/*_${SHORT}_*_greedy 2>/dev/null | head -8
exit $FAIL
