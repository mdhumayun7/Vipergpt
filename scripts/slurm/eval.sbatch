#!/usr/bin/env bash
#SBATCH --job-name=viper-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail

# Usage: sbatch --export=ALL,EXP=repro_table1_refcoco scripts/slurm/eval.sbatch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# module load cuda/12.1 anaconda3
source activate vipergpt

mkdir -p outputs/slurm
: "${EXP:?Set EXP, e.g. repro_table1_refcoco}"
make full EXP="$EXP"
