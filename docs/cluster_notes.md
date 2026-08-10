# Cluster notes

Fill in for the specific HPC being used. Compute nodes have NO internet — all
downloads happen on the login node first.

## Module loads (login + compute)
```
# module load cuda/12.1
# module load anaconda3
```

## Environment
```
conda activate vipergpt
```

## Disk quota / scratch
- pretrained_models path (scratch): `/scratch/<user>/vipergpt/pretrained_models`
- data path (scratch): `/scratch/<user>/vipergpt/data`
- Record `df -h` headroom here BEFORE large downloads. GLIP + BLIP-2 (flan-t5-xxl)
  weights are tens of GB combined.

## Offline mode
- Set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` on compute nodes.
- Validate with `python scripts/verify_offline.py` on the LOGIN node after
  download, before submitting any job.

## GLIP fork build (highest integration risk — R1)
The official repo uses a *modified* GLIP with updated CUDA kernels. Build it on a
LOGIN node (needs internet + a CUDA toolkit matching the torch build):
```bash
export PATH=/usr/local/cuda/bin:$PATH          # or: module load cuda/12.1
conda activate vipergpt
# Obtain the fork bundled as a submodule of cvlab-columbia/viper:
git clone --recurse-submodules https://github.com/cvlab-columbia/viper.git /tmp/viper
cd /tmp/viper/GLIP
python setup.py clean --all build develop --user
python -c "import maskrcnn_benchmark; print('GLIP OK')"
```
Then place GLIP weights + config under `$PRETRAINED_MODEL_PATH/GLIP/`
(`glip_large_model.pth`, `configs/glip_Swin_L.yaml`) as `models/glip.py` expects.
If the build fails against the cluster CUDA, fall back to GroundingDINO for `find`
(record as a deviation) — its box outputs map onto the same `(left,lower,right,upper)`
conversion.

## Manually-downloaded weights (not auto-fetched)
- GLIP-L (`glip_large_model.pth`) → `$PRETRAINED_MODEL_PATH/GLIP/`
- X-VLM checkpoint → `$PRETRAINED_MODEL_PATH/xvlm/` (see models/_xvlm_backbone.py)
- BLIP-2 flan-t5-xxl, MiDaS, Qwen2.5-Coder-7B, Llama-3.1-8B → HF hub (auto), cached
  under `HF_HOME`; pre-download on the login node, then run with `HF_HUB_OFFLINE=1`.

## Node quirks
- (record here: partition names, GPU types/memory, max walltime, etc.)
