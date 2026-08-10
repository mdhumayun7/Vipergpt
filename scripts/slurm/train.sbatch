#!/usr/bin/env bash
#SBATCH --job-name=viper-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
set -euo pipefail

# ViperGPT is training-free; this script is a placeholder kept for the standard
# repo layout (e.g. for any optional fine-tuning experiments under exp/*).
echo "ViperGPT is training-free — no training stage. See docs/implementation_plan.md."
