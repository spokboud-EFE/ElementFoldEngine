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


#python - <<'PY'
#import torch
#print("Torch version:", torch.__version__)
#print("CUDA available?", torch.cuda.is_available())
#print("MPS (Apple Metal) available?", torch.backends.mps.is_available())
#PY


#python - <<'PY'
#from elementfold.rung_controller import RungController, RungIntent
#
#r = RungController(delta=0.5, intent=RungIntent.SEEK, k_target=3)
#
#print("Initial:", r.status())
#for step in range(12):
#    tele = {"κ":0.3, "p½":0.45, "x_mean": step*0.25}
#    ctrl = r.update(tele, {"beta":1.0,"gamma":0.5,"clamp":5.0})
#    print(f"step {step:02d} | ctrl={ctrl} | state={r.status()['phase']}")
#PY


## Studio
#python -m elementfold studio
## In Studio:
#> /mod resonator
#> help
#> init δ=0.5
#> hold
#> tick 5
#> step up 2
#> status

#from elementfold.experience.adapters.base import AdapterRegistry
#run = AdapterRegistry.get("resonator")()
#print(run(None, "init δ=0.5"))
#print(run(None, "hold"))
#print(run(None, "tick 6"))
#print(run(None, "step up 1"))
#print(run(None, "status"))

#(inside studio)
#help                 — show commands
#status               — controller + driver snapshot
#init δ=<value>       — reset and set δ⋆
#hold                 — keep nearest rung
#step up [N]          — cross upward N clicks
#step down [N]        — cross downward N clicks
#delta <value>        — change δ⋆ live
#tick [N]             — run N control ticks
#driver sim|null      — switch driver

