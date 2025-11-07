# ElementFold · __main__.py
# ============================================================
# Package entry point — forwards to the ElementFold CLI.
#
# Quick Usage
# ------------
# 0) Help and all commands
#       python -m elementfold --help
#
# 1) Environment check (Python / Torch / CUDA / MPS)
#       python -m elementfold doctor
#
# 2) Training examples
#       python -m elementfold train --config configs/small.toml \
#            --steps 400 --print-every 100 --out runs/small_01
#
#    Force CPU (smoke test)
#       python -m elementfold train --device cpu --steps 200 --print-every 50 \
#            --out runs/cpu_smoke
#
# 2b) Rung-aware training
#    • Stabilize (default): stay coherent, avoid mid-step crossings
#         python -m elementfold train --steps 600 --print-every 100 --out runs/stable
#    • Hold: lock to nearest rung (optionally target k)
#         python -m elementfold train --rung-intent hold --rung-band 0.006 --out runs/hold
#    • Seek: deliberately cross mid-steps to harvest increments
#         python -m elementfold train --rung-intent seek --rung-band 0.02 \
#            --rung-loss-weight 0.05 --out runs/seek
#
# 3) Train SteeringController (supervised helper)
#       python -m elementfold steering-train --steps 800 --out runs/steering/ctrl.pt
#
# 4) Inference
#       python -m elementfold infer --ckpt runs/small_01/checkpoint.pt --prompt "Hello!"
#       python -m elementfold infer --steps 300 --prompt "Hello world"
#
# 5) Studio interactive REPL
#       python -m elementfold studio
# ============================================================

from .cli import main as _main

if __name__ == "__main__":
    _main()
