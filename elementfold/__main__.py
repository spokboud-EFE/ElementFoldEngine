# ElementFold · __main__.py
# Package entry point: forward to the CLI.
# # ElementFold — Quick usage
#
# # 0) Explore all commands and flags
# python -m elementfold --help
#
# # 1) Environment check (Python/Torch/CUDA/MPS)
# python -m elementfold doctor
#
# # 2) Train with a config file (TOML/JSON) and override a few params at the CLI
# python -m elementfold train --config configs/small.toml --steps 400 --print-every 100 --out runs/small_01
#
# #    Force CPU (smoke test) and save a checkpoint
# python -m elementfold train --device cpu --steps 200 --print-every 50 --out runs/cpu_smoke
#
# # 2b) Rung‑aware training (optional)
# #    Stabilize (default): stay coherent, avoid mid‑step crossings
# python -m elementfold train --steps 600 --print-every 100 --out runs/stable
# #    Hold: lock to the nearest rung (or a specific k) with an acceptance band (defaults to δ⋆/6)
# python -m elementfold train --rung-intent hold --rung-band 0.006 --out runs/hold_nearest
# python -m elementfold train --rung-intent hold --rung-target-k 0 --rung-band 0.006 --out runs/hold_k0
# #    Seek: deliberately cross mid‑steps to harvest increments
# python -m elementfold train --rung-intent seek --rung-band 0.02 --rung-loss-weight 0.05 --out runs/seek
#
# # 3) Train the SteeringController (tiny supervised helper)
# python -m elementfold steering-train --steps 800 --print-every 100 --out runs/steering/ctrl.pt
#
# # 4) Infer from your run (language adapter when --prompt is set)
# python -m elementfold infer --ckpt runs/small_01/checkpoint.pt --prompt "A calm introduction..."
# #    Or quick train‑then‑infer (no checkpoint provided)
# python -m elementfold infer --steps 300 --print-every 50 --prompt "Hello, world"
#
# # 5) Studio: interactive steering REPL in the terminal
# python -m elementfold studio


from .cli import main as _main

if __name__ == "__main__":
    _main()
