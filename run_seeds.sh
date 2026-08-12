#!/bin/bash
#SBATCH --job-name=viper_seeds
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
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "[FATAL] no GPU"; exit 1; }
for S in 1 2 3; do
  echo ">>> SEED $S (T=0.7)"
  python -m vipergpt_repro.eval.codegen_analysis --version refcoco --split testA \
      --max-samples 500 --prompt prompts/api_grounding.prompt \
      --temperature 0.7 --seed $S
done
