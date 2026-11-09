"""Lance DVC en contournant un plantage de Python 3.13 sous Windows.

Sur certaines installations, `logging` appelle asyncio.current_task() et le processus
plante (access violation) dès que DVC écrit un avertissement. Si `dvc` plante chez vous
avec le code 139 / 0xC0000005, utilisez :

    python scripts/dvc_safe.py repro
    python scripts/dvc_safe.py pull
"""
import logging
import sys

logging.logAsyncioTasks = False

from dvc.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
