#!/bin/bash
#SBATCH --job-name=viper_sweepexec
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "[FATAL] no GPU"; exit 1; }
for TAG in 1.5B_base 1.5B_grounding 32B_base 32B_grounding; do
  D=$(ls -1dt outputs/runs/*m1_refcoco_testA_${TAG}_greedy 2>/dev/null | head -1)
  if [ -z "$D" ]; then echo "[SKIP] no run dir for $TAG"; continue; fi
  echo ""
  echo ">>> $TAG  ($D)"
  python -m vipergpt_repro.eval.execute_refcoco --programs "$D/programs.jsonl" \
      --max-samples 500 --tag sweep_${TAG}
done
