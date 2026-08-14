#!/bin/bash
#================================================================
# Milestone 3 — generation for the three-condition comparison.
#
#   C1  api.prompt            released prompt          (have: 0.00%)
#   C2  api_grounding.prompt  return-type contract     (have: 39.20% / 6.38% depth)
#   C3  api_depth.prompt      depth primitives          <- the contribution
#
# Generation only, greedy, so results are comparable to the existing headline
# numbers rather than to the sampled ones. Execution runs afterwards in glip_env.
#
# C1 and C2 are regenerated here even though we have them, because the run
# directory naming changed when the model sweep was added. Cheap, and it keeps all
# three conditions on identical code.
#================================================================
#SBATCH --job-name=viper_m3gen
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
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

echo "node  : $SLURMD_NODENAME"
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }

# RefCOCO+ carries the depth-word queries (closest / farthest / behind), which is
# where the contribution should show. RefCOCO is the control: its spatial queries are
# 2D and should be UNAFFECTED — if they degrade, the depth prompt has traded one
# failure for another.
for DS in refcoco+ refcoco; do
  for P in api_depth.prompt api_grounding.prompt api.prompt; do
    echo ""
    echo ">>> $DS | $P | greedy"
    python -m vipergpt_repro.eval.codegen_analysis \
        --version "$DS" --split testA --max-samples 500 \
        --model "$MODEL" --prompt "prompts/$P" \
        --batch-size 8 --temperature 0.0
  done
done

echo ""
echo "Run directories:"
ls -1dt outputs/runs/*_greedy 2>/dev/null | head -8
