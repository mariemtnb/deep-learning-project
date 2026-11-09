"""Découpage train / val / test par patient, puis export de la table finale.

Entrées : interim/aligned.parquet, interim/official_split.parquet, interim/pneumonia_index.parquet
Sorties : processed/dataset.parquet, processed/pneumonia_index.parquet

Le test officiel NIH est gardé (il est déjà séparé par patient). La validation est tirée
dans le train_val en tirant des PATIENTS, pas des images : un patient a en moyenne 3,6
radiographies et parfois plus de 100, les répartir au hasard ferait fuiter son identité.
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from src.common.config import Config
from src.common.tracking import stage_run

KEEP = ["image", "patient_id", "age_clean", "age_invalid", "age_group", "gender", "view",
        "followup", "is_followup", "width", "height", "aspect_ratio", "finding_labels",
        "is_normal", "n_findings", "is_multilabel", "split", "image_folder", "image_found",
        "has_image", "has_tabular", "has_bbox"]


def assign_split(df: pd.DataFrame, official: pd.DataFrame, val_fraction: float,
                 group_key: str, seed: int) -> pd.DataFrame:
    df = df.merge(official, on="image", how="left", validate="one_to_one")
    df["split"] = df["official_split"].fillna("unlisted")

    tv_groups = np.sort(df.loc[df["split"] == "trainval", group_key].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(tv_groups)
    val_groups = set(tv_groups[: int(len(tv_groups) * val_fraction)])

    is_tv = df["split"] == "trainval"
    df.loc[is_tv & df[group_key].isin(val_groups), "split"] = "val"
    df.loc[df["split"] == "trainval", "split"] = "train"
    return df.drop(columns="official_split")


def run(cfg: Config) -> None:
    s = cfg["split"]
    lbl_cols = [f"lbl_{p}" for p in cfg["preprocess"]["pathologies"]]
    srcs = [cfg.interim("aligned.parquet"), cfg.interim("official_split.parquet"),
            cfg.interim("pneumonia_index.parquet")]
    dst = cfg.processed("dataset.parquet")
    dst_pneu = cfg.processed("pneumonia_index.parquet")
    with stage_run("split", cfg) as rec:
        rec.inputs(*srcs)
        df = assign_split(pd.read_parquet(srcs[0]), pd.read_parquet(srcs[1]),
                          s["val_fraction"], s["group_key"], cfg["seed"])
        df = df[KEEP + lbl_cols].sort_values("image").reset_index(drop=True)
        cfg.path("processed").mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        shutil.copyfile(srcs[2], dst_pneu)
        rec.outputs(dst, dst_pneu)
        counts = df["split"].value_counts().to_dict()
        rec.stats(rows=len(df), **{f"split_{k}": int(v) for k, v in counts.items()},
                  val_patients=int(df.loc[df.split == "val", s["group_key"]].nunique()))
