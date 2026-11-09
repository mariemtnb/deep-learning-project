"""Démonstration d'un échec contrôlé : on corrompt une copie de l'échantillon et on
montre que le pipeline s'arrête au bon contrôle, avec un code de sortie 1.

    python scripts/demo_quality_failure.py --fault leak           # patient dans 2 ensembles
    python scripts/demo_quality_failure.py --fault missing-image  # une radio disparaît
    python scripts/demo_quality_failure.py --fault bad-values     # sexe « X », âge 411, doublon

Rien n'est modifié dans data/ : tout se passe dans build/demo/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "build" / "demo"


def corrupt(fault: str) -> None:
    nih, imgs = DEMO / "nih", DEMO / "nih_images" / "images_001" / "images"
    if fault == "missing-image":
        victim = sorted(imgs.iterdir())[0]
        victim.unlink()
        print(f"-> image supprimée : {victim.name}")
    elif fault == "bad-values":
        csv = nih / "Data_Entry_2017.csv"
        lines = csv.read_text().splitlines()
        cols = lines[1].split(",")
        cols[4], cols[5] = "411", "X"
        lines[1] = ",".join(cols)
        lines.append(lines[2])  # doublon exact
        csv.write_text("\n".join(lines) + "\n")
        print(f"-> ligne {cols[0]} : âge 411, sexe X ; ligne {lines[2].split(',')[0]} dupliquée")
    elif fault == "leak":
        # On déplace UNE image d'un patient du test officiel vers train_val :
        # le patient se retrouve à cheval sur train/val et test.
        test = (nih / "test_list.txt").read_text().split()
        tv = (nih / "train_val_list.txt").read_text().split()
        counts: dict[str, int] = {}
        for name in test:
            counts[name[:8]] = counts.get(name[:8], 0) + 1
        patient = next(p for p, n in counts.items() if n >= 2)
        moved = next(n for n in test if n.startswith(patient))
        test.remove(moved)
        tv.append(moved)
        (nih / "test_list.txt").write_text("\n".join(test) + "\n")
        (nih / "train_val_list.txt").write_text("\n".join(tv) + "\n")
        print(f"-> {moved} déplacée de test vers train_val (patient {int(patient)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=["leak", "missing-image", "bad-values"], default="leak")
    fault = ap.parse_args().fault

    shutil.rmtree(DEMO, ignore_errors=True)
    shutil.copytree(ROOT / "data" / "sample", DEMO)
    corrupt(fault)
    cfg = DEMO / "demo.yaml"
    cfg.write_text(
        "extends: ../../configs/sample.yaml\n"
        "paths:\n"
        "  raw_nih: build/demo/nih\n"
        "  nih_images: build/demo/nih_images\n"
        "  pneumonia_images: build/demo/pneumonia\n"
        "  interim: build/demo/out/interim\n"
        "  processed: build/demo/out/processed\n"
        "  reports: build/demo/out/reports\n"
        "  outputs: build/demo/out/outputs\n"
        "  logs: build/demo/out/logs\n", encoding="utf-8")
    print(f"\n$ python -m src.cli all --config {cfg.relative_to(ROOT).as_posix()}\n")
    code = subprocess.call([sys.executable, "-m", "src.cli", "all", "--config", str(cfg)],
                           cwd=ROOT)
    print(f"\ncode de sortie : {code} ({'arrêt attendu' if code else 'AUCUN ARRÊT ?!'})")
    return 0 if code == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
