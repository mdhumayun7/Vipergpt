#!/bin/bash
#================================================================
# Condition C4 — conditional depth prompt, generation THEN execution.
#
# Tests whether the -10 point non-spatial loss in C3 is over-application. If the
# routing instruction recovers non-spatial accuracy while preserving the depth-query
# gain, the contribution is a strict improvement rather than a trade.
#
# Both stages in one job. Generation needs the `vipergpt` env (torch 2.13) and
# execution needs `glip_env` (torch 2.1.2, required by the GLIP fork), so each stage
# activates its own environment; they never coexist in one interpreter.
#================================================================
#SBATCH --job-name=viper_c4
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00

module load anaconda3-2024.2
module load cuda-12.8
cd /home/mazaveri/hpc-prog/humayun/vipergpt
mkdir -p outputs/slurm
export HF_HOME=/home/mazaveri/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
source "$(conda info --base)/etc/profile.d/conda.sh"

MODEL=Qwen/Qwen2.5-Coder-7B-Instruct

echo "node : $SLURMD_NODENAME"

#---------------------------------------------------------------- generation
conda activate vipergpt || true
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
python -c "import torch,sys; ok=torch.cuda.is_available(); print('CUDA:',ok); sys.exit(0 if ok else 1)" \
  || { echo "[FATAL] no GPU"; exit 1; }

for DS in refcoco+ refcoco; do
  echo ""
  echo ">>> GEN | $DS | api_depth_cond.prompt | greedy"
  python -m vipergpt_repro.eval.codegen_analysis \
      --version "$DS" --split testA --max-samples 500 \
      --model "$MODEL" --prompt prompts/api_depth_cond.prompt \
      --batch-size 8 --temperature 0.0
done
conda deactivate || true

#---------------------------------------------------------------- adoption check
# Before spending an hour on execution, confirm the routing instruction did not
# suppress the primitives entirely. C3 saw 167 / 119 depth_order calls.
python - <<'PYEOF'
import json, glob, collections
for d in sorted(glob.glob("outputs/runs/*_depth_cond_greedy")):
    recs = [json.loads(l) for l in open(d + "/programs.jsonl")]
    c = collections.Counter()
    for r in recs:
        for n in ("depth_order", "is_behind", "compute_depth", "find"):
            c[n] += r["program"].count(n + "(")
    print(f"{d.split('__')[-1]:<46} depth_order={c['depth_order']:<5} "
          f"is_behind={c['is_behind']:<4} find={c['find']}")
PYEOF

#---------------------------------------------------------------- execution
conda activate glip_env || true
export PATH="/home/mazaveri/.conda/envs/glip_env/bin:$PATH"
export PYTHONPATH=$PWD/src

for DS in refcoco+ refcoco; do
  D=$(ls -1dt outputs/runs/*m1_${DS}_testA_7B_depth_cond_greedy 2>/dev/null | head -1)
  [ -z "$D" ] && { echo "[SKIP] no run dir for $DS"; continue; }
  echo ""
  echo ">>> EXEC | $DS  ($D)"
  python -m vipergpt_repro.eval.execute_refcoco --programs "$D/programs.jsonl" \
      --version "$DS" --max-samples 500 --tag c4_${DS}
done

echo ""
echo "Reference — C3 (unconditional depth prompt):"
echo "  RefCOCO+  overall 28.00  spatial 19.15  non-spatial 28.92"
echo "  RefCOCO   overall 36.40  spatial 36.68  non-spatial 36.02"
echo "Reference — C2 (grounding contract):"
echo "  RefCOCO+  overall 30.80  spatial  4.26  non-spatial 33.55"
echo "  RefCOCO   overall 39.20  spatial 34.26  non-spatial 45.97"
