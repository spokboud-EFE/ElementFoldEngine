#!/bin/bash
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# chmod +x miniconda.sh
# ./miniconda.sh
cat > conda_env.yml << 'EOF'
name: elementfold
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  # ---- core Python ----
  - python=3.11

  # ---- PyTorch GPU/CPU ----
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=12.1       # change to 12.4 if your driver requires it

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

  # ---- audio IO ----
  - python-sounddevice      # runtime mic/speaker I/O (uses portaudio)
  - pysoundfile             # read/write wav/flac (libsndfile)
  - portaudio               # native backend for sounddevice
  - libsndfile              # backend for pysoundfile

  # ---- pip bridge ----
  - pip
  - pip:
      - setuptools>=70      # fix for _distutils_hack
      - rich
EOF

# (1) Make sure conda shell hooks are loaded for 'conda activate' in this shell
#eval "$(conda shell.bash hook)"

# (2) Create (first time)
conda env create -f conda_env.yml

## (3) Or update an existing env to match the file
#conda env update -f conda_env.yml --prune
#
## (4) Activate
conda activate elementfold
#
## (5) Sanity check
#python -V
#python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
