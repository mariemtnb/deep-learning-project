"""Construit l'échantillon versionné dans Git (data/sample/), utilisé par la CI.

On prend des patients entiers (pour garder la logique de découpage par patient), dont
au moins un cas avec boîte englobante et un âge aberrant, et on réduit les images à
128x128 en niveaux de gris pour que tout tienne en ~2 Mo.

    python scripts/make_sample.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.config import load_config  # noqa: E402

N_PATIENTS = 40
SIZE = 128
SEED = 7


def main() -> None:
    cfg = load_config(ROOT / "configs" / "params.yaml")
    raw, img_root = cfg.path("raw_nih"), cfg.path("nih_images")
    pneu_root = cfg.path("pneumonia_images")
    out = ROOT / "data" / "sample"
    shutil.rmtree(out, ignore_errors=True)
    (out / "nih").mkdir(parents=True)

    header = (raw / "Data_Entry_2017.csv").read_text().splitlines()[0]
    meta = pd.read_csv(raw / "Data_Entry_2017.csv")
    bbox_lines = (raw / "BBox_List_2017.csv").read_text().splitlines()
    test = set((raw / "test_list.txt").read_text().split())

    # Patients dont toutes les images sont dans images_001, au plus 8 images chacun.
    in_001 = {p.name for p in (img_root / "images_001" / "images").iterdir()}
    meta["in_001"] = meta["Image Index"].isin(in_001)
    per = meta.groupby("Patient ID").agg(n=("in_001", "size"), ok=("in_001", "all"))
    pool = per[(per.ok) & (per.n <= 8)].index.to_numpy()
    rng = np.random.default_rng(SEED)

    bbox_imgs = {line.split(",")[0] for line in bbox_lines[1:]}
    must = set(meta.loc[meta["Image Index"].isin(bbox_imgs) & meta["in_001"], "Patient ID"])
    must = set(sorted(must & set(pool))[:2])
    bad_age = meta.loc[(meta["Patient Age"] > 100) & meta["in_001"], "Patient ID"]
    must |= set(bad_age[bad_age.isin(pool)].head(1))
    test_patients = set(meta.loc[meta["Image Index"].isin(test), "Patient ID"]) & set(pool)
    must |= set(rng.choice(sorted(test_patients), 8, replace=False))
    rest = rng.choice(sorted(set(pool) - must), N_PATIENTS - len(must), replace=False)
    chosen = must | set(rest)

    sub = meta[meta["Patient ID"].isin(chosen)].drop(columns="in_001")
    names = set(sub["Image Index"])
    lines = (raw / "Data_Entry_2017.csv").read_text().splitlines()[1:]
    keep = [ln for ln in lines if ln.split(",")[0] in names]
    (out / "nih" / "Data_Entry_2017.csv").write_text("\n".join([header, *keep]) + "\n")
    (out / "nih" / "BBox_List_2017.csv").write_text(
        "\n".join([bbox_lines[0], *[ln for ln in bbox_lines[1:]
                                    if ln.split(",")[0] in names]]) + "\n")
    for f in ["test_list.txt", "train_val_list.txt"]:
        kept = [x for x in (raw / f).read_text().split() if x in names]
        (out / "nih" / f).write_text("\n".join(kept) + "\n")

    dst = out / "nih_images" / "images_001" / "images"
    dst.mkdir(parents=True)
    for name in sorted(names):
        with Image.open(img_root / "images_001" / "images" / name) as im:
            im.convert("L").resize((SIZE, SIZE), Image.BILINEAR).save(dst / name, optimize=True)

    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            d = out / "pneumonia" / split / cls
            d.mkdir(parents=True)
            for f in sorted((pneu_root / split / cls).glob("*.jpeg"))[:2]:
                with Image.open(f) as im:
                    im.convert("L").resize((SIZE, SIZE)).save(d / f.name, quality=85)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"échantillon : {len(chosen)} patients, {len(names)} images, {size:.1f} Mo")


if __name__ == "__main__":
    main()
