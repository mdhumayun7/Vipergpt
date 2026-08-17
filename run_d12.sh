#!/bin/bash
#================================================================
# D12 ablation — does capping the candidate set help end to end?
#
# The prototype selection study (fig12) showed that ordering 39.5 candidates by
# depth succeeds 29.0% of the time, while ordering 6 succeeds 32.3%. That was
# oracle-style picking, not generated programs. This asks the real question:
# does setting max_detections change C4's accuracy when actual programs run?
#
# Two effects are expected and they pull in opposite directions:
#   + a smaller set is easier for a program to select from
#   - a smaller set sometimes no longer contains the correct box (oracle 83.9 -> 61.3)
# Which dominates is what the experiment settles.
#
# It should also remove the per-candidate CLIP cost that caused the C4 timeouts:
# verify_property over 6 boxes instead of 40.
#
# No generation needed — the same C4 programs are re-executed under each setting,
# so any difference is attributable to the candidate set alone.
#================================================================
#SBATCH --job-name=viper_d12
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

echo "node : $SLURMD_NODENAME"
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }

# MAX_DET is read by execute_refcoco via the environment (see patch below).
for K in 0 12 6 3; do
  for DS in refcoco refcoco+; do
    D=$(ls -1dt outputs/runs/*m1_${DS}_testA_7B_depth_cond_greedy 2>/dev/null | head -1)
    [ -z "$D" ] && { echo "[SKIP] no C4 run dir for $DS"; continue; }
    LBL=$([ "$K" = "0" ] && echo "off" || echo "k$K")
    echo ""
    echo ">>> $DS | max_detections=$K | nms=0.5"
    MAX_DET=$K FIND_NMS=0.5 python -m vipergpt_repro.eval.execute_refcoco \
        --programs "$D/programs.jsonl" --version "$DS" \
        --max-samples 500 --timeout 300 --tag d12_${DS}_${LBL}
  done
done

echo ""
echo "Reference — C4 with no cap (max_detections=0):"
echo "  RefCOCO   overall 42.00  spatial 37.02  non-spatial 48.82"
echo "  RefCOCO+  overall 37.60  spatial 21.28  non-spatial 39.29"
