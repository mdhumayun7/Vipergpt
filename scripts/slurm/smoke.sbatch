#!/usr/bin/env bash
#SBATCH --job-name=viper-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail

# [COMPUTE NODE] No internet here. Weights must already be on scratch.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# module load cuda/12.1 anaconda3   # adjust to your cluster (see docs/cluster_notes.md)
source activate vipergpt

mkdir -p outputs/slurm
make smoke
