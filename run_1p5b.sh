#!/bin/bash
#================================================================
# Model sweep — Qwen2.5-Coder-1.5B, both prompt conditions.
#
# Tests whether the prompt-specification effect is scale-dependent.
# 7B is already measured (baseline 0.00%, grounding 39.20%); this is
# the lower end. Greedy decoding throughout, so results are directly
# comparable to the 7B headline numbers rather than to the sampled ones.
#
# Generation only. Execution runs later in glip_env.
#================================================================
#SBATCH --job-name=viper_1p5b
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:10
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00

module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

MODEL=Qwen/Qwen2.5-Coder-1.5B-Instruct

echo "node   : $SLURMD_NODENAME"
echo "model  : $MODEL"
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }

for P in api.prompt api_grounding.prompt; do
  echo ""
  echo ">>> 1.5B | prompt=$P | RefCOCO/testA | greedy"
  python -m vipergpt_repro.eval.codegen_analysis \
      --version refcoco --split testA --max-samples 500 \
      --model "$MODEL" --prompt "prompts/$P" \
      --batch-size 16 --temperature 0.0
done

echo ""
echo "Run directories produced:"
ls -1dt outputs/runs/*m1_refcoco_testA_greedy 2>/dev/null | head -4
