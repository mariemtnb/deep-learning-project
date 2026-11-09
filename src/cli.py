"""Point d'entrée unique du pipeline.

    python -m src.cli <etape> [--config configs/params.yaml]
    python -m src.cli all --config configs/sample.yaml      # tout, dans l'ordre

Avec DVC, on n'appelle pas ce fichier à la main : `dvc repro` lit dvc.yaml, qui appelle
chaque étape avec ses dépendances et ne relance que ce qui a changé. La commande `all`
sert à la CI et aux machines sans DVC.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys

from src.common.config import load_config


def _script(module: str):
    def run(cfg):
        os.environ["DL_OUTPUTS_DIR"] = str(cfg.path("outputs"))
        runpy.run_module(module, run_name="__main__")
    return run


def _stages() -> dict:
    from src.alignment import align
    from src.analysis import modalities
    from src.ingestion import images, nih_metadata
    from src.monitoring import drift
    from src.preprocessing import clean
    from src.reporting import examples, summary
    from src.splitting import split
    from src.validation import checks

    return {
        "ingest": nih_metadata.run,
        "index_images": images.run,
        "validate_raw": checks.run_raw,
        "preprocess": clean.run,
        "align": align.run,
        "split": split.run,
        "validate": checks.run_processed,
        "analyze": modalities.run,
        "report": summary.run,
        "examples": examples.run,
        "dashboard": _script("src.reporting.dashboard"),
        "drift": drift.run,
    }


EXTRA = {"explanation": "src.reporting.explanation"}


def main(argv: list[str] | None = None) -> int:
    stages = _stages()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*stages, *EXTRA, "all"])
    parser.add_argument("--config", default="configs/params.yaml")
    parser.add_argument("--set-reference", action="store_true",
                        help="drift : enregistre la version courante comme référence")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    from src.validation.checks import QualityGateError
    try:
        if args.stage == "all":
            for fn in stages.values():
                fn(cfg)
        elif args.stage in EXTRA:
            _script(EXTRA[args.stage])(cfg)
        elif args.stage == "drift":
            stages["drift"](cfg, set_reference=args.set_reference)
        else:
            stages[args.stage](cfg)
    except QualityGateError as err:
        print(f"\nPIPELINE ARRÊTÉ : {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
