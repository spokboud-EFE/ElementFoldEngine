# ElementFold · __main__.py
# Package entry point: forward to the CLI.
# Usage:
#   python -m elementfold            # same as `python -m elementfold.cli`
#   python -m elementfold --help
#   python -m elementfold --studio
#   python -m elementfold --steps 400 --seq_len 256

# Train / quick smoke
# python -m elementfold --steps 200 --seq_len 128
#
# Studio (REPL)
# python -m elementfold --studio
#
# Server (API + static UI)
# python -m elementfold.server --host 0.0.0.0 --port 8080
# Then open: http://127.0.0.1:8080/
#
# (Optional) Verify diagnostics
# python -m elementfold.verify (see optional CLI note below)
# python -m elementfold --help

from .cli import main as _main

if __name__ == "__main__":
    _main()
