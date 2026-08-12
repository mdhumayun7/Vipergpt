#!/bin/bash
#SBATCH --job-name=viper_seedexec
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:30
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" || { echo "[FATAL] no GPU"; exit 1; }
for S in 1 2 3; do
  D=$(ls -1dt outputs/runs/*m1_refcoco_testA_t0.7s${S} | head -1)
  echo ">>> SEED $S  ($D)"
  python -m vipergpt_repro.eval.execute_refcoco --programs "$D/programs.jsonl" \
      --max-samples 500 --tag grounding_s${S}
done
