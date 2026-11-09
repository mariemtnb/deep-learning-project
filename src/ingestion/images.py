"""Ingestion de la modalité image : inventaire des radiographies sur disque.

Entrées  : dossiers images_001..012 (NIH) et chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
Sorties  : interim/image_index.parquet      une ligne par fichier NIH trouvé
           interim/image_checks.parquet     dimensions/mode d'un échantillon d'images
           interim/pneumonia_index.parquet  inventaire du jeu pédiatrique

Les images ne sont pas copiées : elles sont immuables et pèsent 45 Go. On produit un
manifeste (nom, dossier, taille en octets) qui sert de référence pour les contrôles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from src.common.config import Config
from src.common.tracking import stage_run


def index_nih(root, folder_glob: str, ext: str) -> pd.DataFrame:
    rows = []
    for sub in sorted(root.glob(folder_glob)):
        img_dir = sub / "images"
        if not img_dir.is_dir():
            continue
        for f in img_dir.iterdir():
            if f.suffix.lower() == ext:
                rows.append((f.name, sub.name, f.stat().st_size))
    return pd.DataFrame(rows, columns=["image", "image_folder", "bytes"])


def check_sample(root, index: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Ouvre n images au hasard (lecture de l'en-tête seulement) : taille et mode."""
    if index.empty:
        return pd.DataFrame(columns=["image", "width_px", "height_px", "mode"])
    take = index.sample(n=min(n, len(index)), random_state=seed)
    rows = []
    for img, folder in zip(take["image"], take["image_folder"], strict=True):
        with Image.open(root / folder / "images" / img) as im:
            rows.append((img, im.width, im.height, im.mode))
    return pd.DataFrame(rows, columns=["image", "width_px", "height_px", "mode"])


def index_pneumonia(root, ext: str) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            d = root / split / cls
            if d.is_dir():
                rows += [(f"{split}/{cls}/{f.name}", split, cls, f.stat().st_size)
                         for f in d.iterdir() if f.suffix.lower() == ext]
    return pd.DataFrame(rows, columns=["path", "split", "label", "bytes"])


def run(cfg: Config) -> None:
    img = cfg["images"]
    nih_root, pneu_root = cfg.path("nih_images"), cfg.path("pneumonia_images")

    with stage_run("index_images", cfg) as rec:
        if not nih_root.is_dir():
            raise FileNotFoundError(
                f"Dossier d'images introuvable : {nih_root}. "
                "Téléchargez le jeu NIH (voir README) ou corrigez paths.nih_images.")
        index = index_nih(nih_root, img["folder_glob"], img["extension"])
        checks = check_sample(nih_root, index, img["check_sample"], cfg["seed"])
        pneu = index_pneumonia(pneu_root, img["pneumonia_extension"])

        cfg.path("interim").mkdir(parents=True, exist_ok=True)
        outs = [cfg.interim("image_index.parquet"), cfg.interim("image_checks.parquet"),
                cfg.interim("pneumonia_index.parquet")]
        index.to_parquet(outs[0], index=False)
        checks.to_parquet(outs[1], index=False)
        pneu.to_parquet(outs[2], index=False)
        rec.outputs(*outs)
        rec.stats(nih_images=len(index), nih_folders=int(index["image_folder"].nunique()),
                  nih_gb=round(float(np.sum(index["bytes"])) / 1e9, 2),
                  checked=len(checks), pneumonia_images=len(pneu))
