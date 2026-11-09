"""Ingestion de la modalité tabulaire : métadonnées cliniques NIH ChestX-ray14.

Entrées  : data/raw/nih/{Data_Entry_2017.csv, BBox_List_2017.csv, test_list.txt, train_val_list.txt}
Sorties  : interim/metadata_raw.parquet, interim/bbox.parquet, interim/official_split.parquet

On ne fait ici que lire et renommer : aucune valeur n'est corrigée (c'est le rôle de
preprocessing), pour que la validation brute voie les données telles qu'elles arrivent.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.tracking import stage_run

# Le CSV d'origine a un en-tête mal formé : « OriginalImage[Width,Height] » est coupé
# en deux colonnes par la virgule, et une virgule finale crée une colonne vide.
# On renomme donc par position, ce qui est plus sûr que par nom.
METADATA_COLUMNS = [
    "image", "finding_labels", "followup", "patient_id", "age", "gender", "view",
    "width", "height", "spacing_x", "spacing_y",
]
BBOX_COLUMNS = ["image", "bbox_label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]


def read_metadata(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    if df.shape[1] != len(METADATA_COLUMNS):
        raise ValueError(f"{path.name}: {df.shape[1]} colonnes, {len(METADATA_COLUMNS)} attendues")
    df.columns = METADATA_COLUMNS
    return df


def read_bbox(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.columns = BBOX_COLUMNS
    return df


def read_official_split(test_path, trainval_path) -> pd.DataFrame:
    test = test_path.read_text().split()
    trainval = trainval_path.read_text().split()
    return pd.DataFrame({
        "image": test + trainval,
        "official_split": ["test"] * len(test) + ["trainval"] * len(trainval),
    })


def run(cfg: Config) -> None:
    raw = cfg.path("raw_nih")
    ing = cfg["ingest"]
    src_meta, src_bbox = raw / ing["metadata_file"], raw / ing["bbox_file"]
    src_test, src_tv = raw / ing["test_list"], raw / ing["trainval_list"]

    with stage_run("ingest", cfg) as rec:
        rec.inputs(src_meta, src_bbox, src_test, src_tv)
        meta = read_metadata(src_meta)
        bbox = read_bbox(src_bbox)
        split = read_official_split(src_test, src_tv)

        out_dir = cfg.path("interim")
        out_dir.mkdir(parents=True, exist_ok=True)
        outs = [cfg.interim("metadata_raw.parquet"), cfg.interim("bbox.parquet"),
                cfg.interim("official_split.parquet")]
        meta.to_parquet(outs[0], index=False)
        bbox.to_parquet(outs[1], index=False)
        split.to_parquet(outs[2], index=False)
        rec.outputs(*outs)
        rec.stats(metadata_rows=len(meta), patients=int(meta["patient_id"].nunique()),
                  bbox_rows=len(bbox), official_test=int((split.official_split == "test").sum()),
                  official_trainval=int((split.official_split == "trainval").sum()))
