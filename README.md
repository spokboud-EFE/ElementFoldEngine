# ElementFoldEngine -- EFE
##### Author: Sam Blouin

Proving that coherence itself can be engineered as the new foundation of intelligence and design.

## Usage
```bash
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
```

This work is publicly available exclusively for non-profit organizations.