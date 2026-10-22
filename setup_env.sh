#!/bin/bash
# Source this at the start of EVERY new job:
#   source setup_env.sh   (set NLA_WORK_DIR first if your checkpoints/envs live elsewhere)

# --- paths ---
NLA_WORK_DIR="${NLA_WORK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"  # dir holding checkpoints, data, runs/
export BASE_DIR=$NLA_WORK_DIR/miles_build_v2
export NLA_DIR=$NLA_WORK_DIR/natural_language_autoencoders
export CONSTRAINTS=$BASE_DIR/pip-constraints.txt

# --- pinned commits (the set that actually works together) ---
export SGLANG_COMMIT=1519acf37c23f2189adb93f57ca9cd2db1bebf18   # v0.5.10, has DumperConfig
export MEGATRON_COMMIT=3714d81d418c9f1bca4594fc35f9e8289f652862
export MILES_COMMIT=$(cut -d'@' -f2 $NLA_DIR/nla/miles_patches/UPSTREAM_PIN 2>/dev/null)  # 051cd15

# --- env location on /work (survives job restarts) ---
export MAMBA_ROOT_PREFIX=$NLA_WORK_DIR/microtools/root

# --- reconnect micromamba + activate the /work env ---
if ! command -v micromamba &>/dev/null; then
  echo "micromamba binary missing — reinstall it:"
  echo "  yes '' | \"\${SHELL}\" <(curl -L micro.mamba.pm/install.sh) && source ~/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)" 2>/dev/null
micromamba activate $MAMBA_ROOT_PREFIX/envs/miles
export CUDA_HOME="$CONDA_PREFIX"

echo "env: $(which python)"
python -c "import torch,sglang,transformers; print('torch',torch.__version__,'sglang',sglang.__version__,'transformers',transformers.__version__)" 2>/dev/null \
  || echo "env not importable — may need rebuild"

# flashinfer JIT needs to link against the CUDA driver stub (fixes "cannot find -lcuda")
export LIBRARY_PATH=$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH

# sanity: is the sglang b64 transport patch still in place?
grep -q "input_embeds_b64_bf16" $BASE_DIR/sglang/python/sglang/srt/entrypoints/http_server.py \
  && echo "sglang NLA patch: OK" \
  || echo "WARNING: sglang NLA transport patch MISSING — apply sglang_nla_all.patch"
