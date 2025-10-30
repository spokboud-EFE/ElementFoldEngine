#!/bin/
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
# chmod +x miniconda.sh
# ./miniconda.sh
cat > conda_env.yml <<'EOF'
name: elementfold
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python
  - pip
  - numpy
  - scipy
  - pytorch
  - torchvision
  - torchaudio
  - matplotlib
  - pandas
  - numba
  - sympy
  - pytest
  - black
  - isort
EOF
conda env create -f conda_env.yml
conda activate elementfold