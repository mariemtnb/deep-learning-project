"""Nettoyage et création de variables sur la table clinique.

Entrée  : interim/metadata_raw.parquet
Sortie  : interim/metadata_clean.parquet

- âges impossibles (hors [age_min, age_max]) : marqués puis remplacés par la médiane
  des âges valides (16 cas sur 112 120, ex. 148 ou 411 ans) ;
- sexe et vue normalisés (majuscules, espaces retirés) ;
- `finding_labels` éclaté en 14 colonnes binaires + variables dérivées.
"""
from __future__ import annotations

import pandas as pd

from src.common.config import Config
from src.common.tracking import stage_run


def clean(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_invalid"] = ~df["age"].between(p["age_min"], p["age_max"])
    median = int(df.loc[~df["age_invalid"], "age"].median())
    df["age_clean"] = df["age"].where(~df["age_invalid"], median).astype(int)

    df["gender"] = df["gender"].astype(str).str.upper().str.strip()
    df["view"] = df["view"].astype(str).str.upper().str.strip()
    for c in ["width", "height", "followup"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["spacing_x", "spacing_y"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    labels = df["finding_labels"].fillna("").str.split("|")
    label_sets = labels.apply(set)
    for name in p["pathologies"]:
        df[f"lbl_{name}"] = label_sets.apply(lambda s, n=name: int(n in s))
    lbl_cols = [f"lbl_{n}" for n in p["pathologies"]]
    df["is_normal"] = (df["finding_labels"] == "No Finding").astype(int)
    df["n_findings"] = df[lbl_cols].sum(axis=1)
    df["is_multilabel"] = (df["n_findings"] > 1).astype(int)
    df["age_group"] = pd.cut(df["age_clean"], bins=p["age_bins"], labels=p["age_labels"],
                             right=True).astype(str)
    df["aspect_ratio"] = (df["width"] / df["height"]).round(3)
    df["is_followup"] = (df["followup"] > 0).astype(int)
    return df


def run(cfg: Config) -> None:
    p = cfg["preprocess"]
    src = cfg.interim("metadata_raw.parquet")
    dst = cfg.interim("metadata_clean.parquet")
    with stage_run("preprocess", cfg) as rec:
        rec.inputs(src)
        df = pd.read_parquet(src)
        df = add_features(clean(df, p), p)
        df.to_parquet(dst, index=False)
        rec.outputs(dst)
        rec.stats(rows=len(df), ages_repaired=int(df["age_invalid"].sum()),
                  normal=int(df["is_normal"].sum()),
                  multilabel=int(df["is_multilabel"].sum()))
