#!/bin/bash
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# chmod +x miniconda.sh
# ./miniconda.sh
cat > conda_env.yml <<'EOF'
name: elementfold
channels:
  - pytorch         # official PyTorch builds
  - conda-forge     # wide ecosystem
  - nvidia
  - defaults
dependencies:
  # ---- core Python ----
  - python=3.11

#  # ---- PyTorch stack (CPU build) ----
#  - pytorch
#  - torchvision
#  - torchaudio
#  - cpuonly          # ensures the CPU variant is selected

  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1    # or 12.4 depending on your driver


  # ---- numerics / utils ----
  - numpy
  - scipy
  - numba
  - pandas
  - matplotlib
  - pillow
  - sympy

  # ---- dev & QA ----
  - pytest
  - black
  - isort

  # ---- audio IO (from conda-forge; avoids pip builds) ----
  - python-sounddevice   # sounddevice bindings (depends on portaudio)
  - pysoundfile          # convenient wav/flac loader (libsndfile)
  - portaudio            # runtime library for sounddevice
  - libsndfile           # runtime library for pysoundfile

  - pip
  - pip:
      # - some-pypi-only-package

EOF

# (1) Make sure conda shell hooks are loaded for 'conda activate' in this shell
#eval "$(conda shell.bash hook)"

# (2) Create (first time)
conda env create -f conda_env.yml

## (3) Or update an existing env to match the file
#conda env update -f conda_env.yml --prune
#
## (4) Activate
#conda activate elementfold
#
## (5) Sanity check
#python -V
#python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

# The readme file in engine directory tells engine usage