#!/bin/bash
#================================================================
# SLURM DIRECTIVES
#
#   Submit with:   sbatch run_all.sh
#
#   IMPORTANT: outputs/slurm/ must exist BEFORE submitting, or SLURM
#   silently discards the job with no .out and no .err. One time:
#       mkdir -p outputs/slurm
#
#   This script is self-contained. If the compute node turns out to
#   have internet, it downloads whatever is missing itself. If not,
#   it fails fast and tells you exactly what to fetch on the login node.
#================================================================
#SBATCH --job-name=viper_all
#SBATCH --partition=gpu
#SBATCH --output=outputs/slurm/%x_%j.out
#SBATCH --error=outputs/slurm/%x_%j.err
#SBATCH --gres=shard:25
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#================================================================
# LOAD MODULES
#================================================================
module load anaconda3-2024.2
module load cuda-12.8
#================================================================
# ENVIRONMENT SETUP
#================================================================
set -uo pipefail          # NOT -e: stages are collected, not fatal

PROJECT="/home/mazaveri/hpc-prog/humayun/vipergpt"
STORE="/home/mazaveri/hpc-prog/humayun/vipergpt_store"

export HF_HOME=/home/mazaveri/hf_cache
export DATA_PATH="$STORE/data"
export PRETRAINED_MODEL_PATH="$STORE/pretrained_models"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1        # so .out streams live instead of buffering
CODE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"

echo ">>> Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh" || true
conda activate vipergpt || true
# conda activate can fail silently in a batch shell - force the env onto PATH.
export PATH="/home/mazaveri/.conda/envs/vipergpt/bin:$PATH"
export CONDA_PREFIX="/home/mazaveri/.conda/envs/vipergpt"
# Stop the broken ~/.local torch (python3.11) from shadowing the env.
export PYTHONNOUSERSITE=1
echo "[OK] Python : $(which python)"
echo "[OK] Version: $(python --version 2>&1)"

cd "$PROJECT" || { echo "[FATAL] cannot cd to $PROJECT"; exit 1; }
mkdir -p outputs/slurm outputs/runs results/logs

# Track every stage; print a table at the end.
declare -A ST
for s in preflight data models verify tests smoke m1_refcoco m1_refcocoplus; do ST[$s]=SKIP; done

#================================================================
# SYSTEM INFO
#================================================================
echo ""
echo "============================================================"
echo "   VIPERGPT REPRODUCTION — FULL PIPELINE"
echo "   Scope: RefCOCO / RefCOCO+ (Table 1)"
echo "============================================================"
echo "Date       : $(date)"
echo "Node       : $SLURMD_NODENAME"
echo "Job ID     : $SLURM_JOB_ID"
echo "Job Name   : $SLURM_JOB_NAME"
echo "Project    : $PROJECT"
echo "Data       : $DATA_PATH"
echo "HF cache   : $HF_HOME"
echo "Git SHA    : $(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "Git dirty  : $(git status --porcelain 2>/dev/null | wc -l) file(s)"
echo "Stdout     : outputs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
echo "Stderr     : outputs/slurm/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
echo "============================================================"
echo ""

nvidia-smi
echo ""

python -c "
import torch
print('PyTorch    :', torch.__version__)
print('CUDA       :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU        :', torch.cuda.get_device_name(0))
    print('VRAM       :', round(torch.cuda.get_device_properties(0).total_memory/1e9,1), 'GB')
    print('Capability :', torch.cuda.get_device_capability(0))
print('============================================================')
"

#================================================================
# PHASE 0 — PREFLIGHT
#================================================================
echo ""
echo ">>> PHASE 0: PREFLIGHT"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"

FATAL=0

# The python/pip mismatch that bit us before — catch it here, not 3 hours in.
PY_PREFIX=$(python -c 'import sys; print(sys.prefix)')
if [[ "$PY_PREFIX" != *vipergpt* ]]; then
    echo "[FATAL] python is $PY_PREFIX, not the vipergpt env"
    FATAL=1
else
    echo "[  OK  ] python prefix: $PY_PREFIX"
fi

python -c "import vipergpt_repro" 2>/dev/null \
  && echo "[  OK  ] package importable" \
  || { echo "[FATAL] vipergpt_repro not importable (python -m pip install -e .)"; FATAL=1; }

python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
  && echo "[  OK  ] GPU visible" \
  || { echo "[FATAL] no GPU on this node"; FATAL=1; }

# Compute nodes normally have no internet. Probe rather than assume — the
# result decides whether missing files can be fetched here or must be
# prepared on the login node.
if timeout 10 python -c "import socket; socket.create_connection(('huggingface.co',443),8)" 2>/dev/null; then
    HAS_NET=1
    echo "[ NOTE ] compute node HAS internet — missing files will be downloaded here"
else
    HAS_NET=0
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[ NOTE ] compute node has NO internet — offline mode enabled"
fi

echo "[ INFO ] disk free: $(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')G"

if (( FATAL == 1 )); then
    echo ""
    echo "[FATAL] preflight failed — aborting before wasting the allocation."
    ST[preflight]=FAIL
    exit 1
fi
ST[preflight]=OK
echo ">>> PHASE 0 DONE: $(date)"

#================================================================
# PHASE 1 — DATA CHECK  (no download: compute node is offline)
#================================================================
echo ""
echo ">>> PHASE 1: DATASETS"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"

if [[ ! -f "$DATA_PATH/refcoco/refcoco/instances.json" \
   || ! -f "$DATA_PATH/refcoco/refcoco+/instances.json" ]]; then
    if (( HAS_NET == 1 )); then
        echo "[ .... ] annotations missing — downloading"
        bash scripts/download_data.sh
    else
        echo "[ FAIL ] annotations missing and this node is offline."
        echo "         On the LOGIN node run: bash scripts/download_data.sh"
    fi
fi

python - <<'PY'
import glob, json, os, pickle, sys
root = os.path.join(os.environ["DATA_PATH"], "refcoco")
bad = False
for v in ("refcoco", "refcoco+"):
    d = os.path.join(root, v)
    try:
        with open(os.path.join(d, "instances.json")) as f:
            inst = json.load(f)
        with open(sorted(glob.glob(os.path.join(d, "refs(*).p")))[0], "rb") as f:
            refs = pickle.load(f)
        sents = sum(len(r["sentences"]) for r in refs)
        print(f"[  OK  ] {v}: {len(inst['images'])} imgs, {len(refs)} refs, {sents} sentences")
        print(f"         splits: {sorted({r.get('split') for r in refs})}")
    except Exception as e:
        print(f"[ FAIL ] {v}: {type(e).__name__}: {e}")
        print( "         Download on the LOGIN node: bash scripts/download_data.sh")
        bad = True
sys.exit(1 if bad else 0)
PY
[[ $? -eq 0 ]] && ST[data]=OK || { ST[data]=FAIL; echo "[FATAL] data missing"; }
echo ">>> PHASE 1 DONE: $(date)"

#================================================================
# PHASE 2 — MODEL CHECK
#================================================================
echo ""
echo ">>> PHASE 2: MODEL WEIGHTS"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"

CACHED=$(python - <<PY
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
    p = Path(snapshot_download("$CODE_MODEL", local_files_only=True))
    print(len([f for f in p.rglob("*") if f.suffix in {".safetensors",".bin"} and f.stat().st_size > 1_000_000]))
except Exception:
    print(0)
PY
)
if [[ "$CACHED" == "0" ]]; then
    if (( HAS_NET == 1 )); then
        echo "[ .... ] $CODE_MODEL missing — downloading (~15 GB, this will take hours)"
        hf download "$CODE_MODEL"
    else
        echo "[ FAIL ] $CODE_MODEL missing and this node is offline."
        echo "         On the LOGIN node run: nohup hf download $CODE_MODEL > ~/qwen.log 2>&1 &"
    fi
fi

python - <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
REPO = "Qwen/Qwen2.5-Coder-7B-Instruct"
try:
    p = Path(snapshot_download(REPO, local_files_only=True))
except Exception as e:
    print(f"[ FAIL ] {REPO} not in cache: {type(e).__name__}")
    print( "         Download on the LOGIN node: hf download " + REPO)
    sys.exit(1)
shards = [f for f in p.rglob("*") if f.suffix in {".safetensors",".bin"} and f.stat().st_size > 1_000_000]
if not shards:
    print(f"[ FAIL ] {REPO}: config present but NO weight shards — truncated download")
    sys.exit(1)
print(f"[  OK  ] {REPO}: {len(shards)} shards, {sum(f.stat().st_size for f in shards)/1e9:.1f} GB")
PY
[[ $? -eq 0 ]] && ST[models]=OK || ST[models]=FAIL

echo ""
echo "  Milestone 2 modules (not needed for this pipeline):"
[[ -f "$PRETRAINED_MODEL_PATH/GLIP/checkpoints/glip_large_model.pth" ]] \
  && echo "[  OK  ] GLIP checkpoint" || echo "[BLOCK ] GLIP checkpoint absent"
grep -q "NotImplementedError" src/vipergpt_repro/models/_xvlm_backbone.py 2>/dev/null \
  && echo "[BLOCK ] _xvlm_backbone.py still a stub"
echo ">>> PHASE 2 DONE: $(date)"

# Hard gate: no data or no model means nothing below can run.
if [[ "${ST[data]}" != "OK" || "${ST[models]}" != "OK" ]]; then
    echo ""
    echo "[FATAL] prerequisites missing — fix on the LOGIN node, then resubmit."
    echo "============================================================"
    exit 1
fi

#================================================================
# PHASE 3 — OFFLINE VERIFICATION
#================================================================
echo ""
echo ">>> PHASE 3: OFFLINE VERIFICATION"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"
python scripts/verify_offline.py --skip-weights
[[ $? -eq 0 ]] && ST[verify]=OK || ST[verify]=FAIL
echo ">>> PHASE 3 DONE: $(date)"

#================================================================
# PHASE 4 — TESTS + SMOKE
#================================================================
echo ""
echo ">>> PHASE 4: UNIT TESTS + SMOKE"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"
python -m pytest -m "not slow" -q
[[ $? -eq 0 ]] && ST[tests]=OK || ST[tests]=FAIL

echo ""
python -m vipergpt_repro.pipeline.run_smoke
[[ $? -eq 0 ]] && ST[smoke]=OK || ST[smoke]=FAIL
echo ">>> PHASE 4 DONE: $(date)"

#================================================================
# PHASE 5 — MILESTONE 1: REFCOCO (spatial relations ALLOWED)
#================================================================
echo ""
echo ">>> PHASE 5: M1 ANALYSIS — RefCOCO/testA (spatial allowed)"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"
python -m vipergpt_repro.eval.codegen_analysis \
    --version refcoco --split testA --max-samples 500 --batch-size 8
[[ $? -eq 0 ]] && ST[m1_refcoco]=OK || ST[m1_refcoco]=FAIL
echo ">>> PHASE 5 DONE: $(date)"

#================================================================
# PHASE 6 — MILESTONE 1: REFCOCO+ (spatial relations EXCLUDED — control)
#================================================================
echo ""
echo ">>> PHASE 6: M1 ANALYSIS — RefCOCO+/testA (spatial excluded, control)"
echo ">>> Start: $(date)"
echo "------------------------------------------------------------"
python -m vipergpt_repro.eval.codegen_analysis \
    --version refcoco+ --split testA --max-samples 500 --batch-size 8
[[ $? -eq 0 ]] && ST[m1_refcocoplus]=OK || ST[m1_refcocoplus]=FAIL
echo ">>> PHASE 6 DONE: $(date)"

#================================================================
# SUMMARY
#================================================================
echo ""
echo "============================================================"
echo "   JOB SUMMARY"
echo "============================================================"
printf "  %-16s %s\n" "STAGE" "RESULT"
printf "  %-16s %s\n" "-----" "------"
FAILED=0
for s in preflight data models verify tests smoke m1_refcoco m1_refcocoplus; do
    printf "  %-16s %s\n" "$s" "${ST[$s]}"
    [[ "${ST[$s]}" == "FAIL" ]] && FAILED=1
done

echo ""
echo "  Run directories:"
ls -1dt outputs/runs/*m1_* 2>/dev/null | head -4 | sed 's/^/    /'

LATEST=$(ls -1dt outputs/runs/*m1_refcoco_* 2>/dev/null | head -1)
if [[ -n "$LATEST" && -f "$LATEST/analysis.md" ]]; then
    echo ""
    echo "  RESULT TABLE (RefCOCO):"
    sed -n '/^| Subset/,/^$/p' "$LATEST/analysis.md" | sed 's/^/    /'
fi

echo ""
echo "  Elapsed : $SECONDS s"
echo "  Finished: $(date)"
echo "============================================================"

exit $FAILED
