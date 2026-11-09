"""Alignement des modalités : chaque ligne clinique est appariée à son image.

Entrées : interim/metadata_clean.parquet, interim/image_index.parquet, interim/bbox.parquet
Sortie  : interim/aligned.parquet

La clé de jointure est le nom de fichier (`image`), qui est aussi la clé primaire du
CSV. Il n'y a pas d'horodatage : les deux modalités sont appariées échantillon par
échantillon. Une absence n'est jamais silencieuse : `has_image` / `has_tabular` forment
le masque de modalités que lira le futur DataLoader de fusion.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.tracking import stage_run

TABULAR_KEYS = ["age_clean", "gender", "view", "followup"]


def align(meta: pd.DataFrame, index: pd.DataFrame, bbox: pd.DataFrame) -> pd.DataFrame:
    df = meta.merge(index[["image", "image_folder"]], on="image", how="left",
                    validate="one_to_one")
    df["has_image"] = df["image_folder"].notna().astype(int)
    df["has_tabular"] = df[TABULAR_KEYS].notna().all(axis=1).astype(int)
    df["has_bbox"] = df["image"].isin(set(bbox["image"])).astype(int)
    # Ancien nom conservé : l'interface et le tableau de bord le lisent.
    df["image_found"] = df["has_image"]
    return df


def run(cfg: Config) -> None:
    srcs = [cfg.interim("metadata_clean.parquet"), cfg.interim("image_index.parquet"),
            cfg.interim("bbox.parquet")]
    dst = cfg.interim("aligned.parquet")
    with stage_run("align", cfg) as rec:
        rec.inputs(*srcs)
        meta, index, bbox = (pd.read_parquet(s) for s in srcs)
        df = align(meta, index, bbox)
        orphan_images = len(set(index["image"]) - set(meta["image"]))
        df.to_parquet(dst, index=False)
        rec.outputs(dst)
        rec.stats(rows=len(df), with_image=int(df["has_image"].sum()),
                  missing_image=int((1 - df["has_image"]).sum()),
                  orphan_images=orphan_images, with_bbox=int(df["has_bbox"].sum()))
