#!/bin/bash
# The code uses this particular home use hardware, "Jetson Orin 64gb Developer Kit Jetpack 6.2, arm64 L4T"
# wget repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh
# chmod +x miniconda.sh
# ./miniconda.sh
cat > conda_env.yml << 'EOF'
name: elementfold
channels:
  - conda-forge
  - defaults
dependencies:
  # ---- Core Python ----
  - python=3.10

  # ---- Numerics / Utilities ----
  - numpy<2
  - scipy
  - pandas
  - numba
  - matplotlib
  - pillow
  - sympy

  # ---- Development & QA ----
  - pytest
  - black
  - isort

  # ---- Audio / IO ----
  - pysoundfile        # correct package (provides soundfile module)
  - portaudio          # backend for sounddevice
  - python-sounddevice # mic/speaker I/O
  - rich               # colored terminal UI

  # ---- TOML parser (for configs on Python <3.11) ----
  - tomli

  # ---- pip bridge ----
  - pip
  - pip:
      - setuptools>=70
EOF

sudo apt-get -y update
sudo apt-get install -y python3-pip libopenblas-dev
wget https://developer.download.nvidia.com/compute/cusparselt/0.7.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo dpkg -i cusparselt-local-tegra-repo-ubuntu2204-0.7.1_1.0-1_arm64.deb
sudo cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.7.1/cusparselt-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
wget https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl


# Make sure conda shell hooks are loaded for 'conda activate' in this shell
#eval "$(conda shell.bash hook)"
conda env create -f conda_env.yml

# Or update an existing env to match the file
# conda env update -f conda_env.yml --prune
conda activate elementfold
python -V
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"
