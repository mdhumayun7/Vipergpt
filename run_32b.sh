#!/bin/bash
#================================================================
# Model sweep — Qwen2.5-Coder-32B, both prompt conditions.
#
# Upper end of the scale sweep. 65.5 GB in bf16, so this needs most of
# one H100: ~75 shards of 94. It will sit in the queue until a node
# drains, which is fine — sbatch starts it unattended.
#
# Generation only. Execution runs later in glip_env.
#================================================================
#SBATCH --job-name=viper_32b
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:75
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00

module load anaconda3-2024.2
module load cuda-12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export PYTHONNOUSERSITE=1
cd /home/mazaveri/hpc-prog/humayun/vipergpt
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1

MODEL=Qwen/Qwen2.5-Coder-32B-Instruct

echo "node   : $SLURMD_NODENAME"
echo "model  : $MODEL"
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok, torch.cuda.get_device_name(0) if ok else ''); print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9,1),'GB' if ok else ''); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }

# 65.5 GB of weights plus activations. Batch size is halved relative to 7B.
for P in api.prompt api_grounding.prompt; do
  echo ""
  echo ">>> 32B | prompt=$P | RefCOCO/testA | greedy"
  python -m vipergpt_repro.eval.codegen_analysis \
      --version refcoco --split testA --max-samples 500 \
      --model "$MODEL" --prompt "prompts/$P" \
      --batch-size 4 --temperature 0.0
done

echo ""
echo "Run directories produced:"
ls -1dt outputs/runs/*m1_refcoco_testA_greedy 2>/dev/null | head -4
