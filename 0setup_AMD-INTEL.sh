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
python -m elementfold --help
# 1) Environment check
#python -m elementfold doctor
#
## 2) Train with a config file (TOML) and override a couple params
#python -m elementfold train --config configs/small.toml --steps 400 --print-every 100 --out runs/small_01
#
## 3) Train the steering controller
#python -m elementfold steering-train --steps 800 --print-every 100 --out runs/steering/ctrl.pt
#
## 4) Infer from your run
#python -m elementfold infer --ckpt runs/test1/checkpoint.pt --prompt "A calm introduction..."


#python - <<'PY'
#import torch
#print("Torch version:", torch.__version__)
#print("CUDA available?", torch.cuda.is_available())
#print("MPS (Apple Metal) available?", torch.backends.mps.is_available())
#PY
