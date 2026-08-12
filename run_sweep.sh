#!/bin/bash
#SBATCH --job-name=viper_sweep
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
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
python sweep_threshold.py --n 200 --start 200
