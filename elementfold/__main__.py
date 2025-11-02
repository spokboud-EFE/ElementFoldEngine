# ElementFold · __main__.py
# Package entry point: forward to the CLI.
# Usage:
#   python -m engine.elementfold            # same as `python -m engine.elementfold.cli`
#   python -m engine.elementfold --help
#   python -m engine.elementfold --studio
#   python -m engine.elementfold --steps 400 --seq_len 256

# Train / quick smoke
# python -m engine.elementfold --steps 200 --seq_len 128
#
# Studio (REPL)
# python -m engine.elementfold --studio
#
# Server (API + static UI)
# python -m engine.elementfold.server --host 0.0.0.0 --port 8080
# Then open: http://127.0.0.1:8080/
#
# (Optional) Verify diagnostics
# python -m engine.elementfold.verify (see optional CLI note below)

from .cli import main as _main

if __name__ == "__main__":
    _main()
