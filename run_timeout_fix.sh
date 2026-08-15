#!/bin/bash
#SBATCH --job-name=viper_tofix
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
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
# Only RefCOCO+ needs this: RefCOCO had 0 timeouts in every condition. All three
# RefCOCO+ conditions are re-run at the same timeout so the comparison stays fair.
for P in depth_cond depth grounding; do
  D=$(ls -1dt outputs/runs/*m1_refcoco+_testA_7B_${P}_greedy 2>/dev/null | head -1)
  [ -z "$D" ] && { echo "[SKIP] $P"; continue; }
  echo ""
  echo ">>> refcoco+ | $P | timeout=300"
  python -m vipergpt_repro.eval.execute_refcoco --programs "$D/programs.jsonl" \
      --version refcoco+ --max-samples 500 --timeout 300 --tag t300_${P}
done
