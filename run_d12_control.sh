#!/bin/bash
#SBATCH --job-name=viper_d12ctl
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8 --mem=64G --time=04:00:00
module load anaconda3-2024.2; module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH" PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache PYTHONPATH=$PWD/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || exit 1

# C2 has NO depth primitives. If capping to k=3 lifts C2 as much as it lifts C4,
# the gain is the candidate set, not the depth reasoning.
for K in 0 3; do
  for P in grounding depth_cond; do
    D=$(ls -1dt outputs/runs/*m1_refcoco_testA_7B_${P}_greedy | head -1)
    echo ""; echo ">>> refcoco | $P | k=$K"
    MAX_DET=$K FIND_NMS=0.5 python -m vipergpt_repro.eval.execute_refcoco \
        --programs "$D/programs.jsonl" --version refcoco \
        --max-samples 500 --timeout 300 --tag ctl_${P}_k${K}
  done
done
