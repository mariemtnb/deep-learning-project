"""Mesures qui servent à choisir la stratégie de fusion (Partie B du rapport).

Entrée : processed/dataset.parquet
Sortie : reports/modality_analysis.json

Questions traitées :
1. Que vaut la modalité tabulaire seule ? (régression logistique par pathologie,
   entraînée sur train, AUC mesurée sur val)
2. La vue (AP/PA) et l'âge modifient-ils la prévalence des pathologies ? Si oui,
   l'information clinique doit pouvoir moduler la lecture de l'image (argument pour FiLM).
3. Volumes et déséquilibre : nombre de positifs par classe dans train, poids pos_weight.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.common.config import Config
from src.common.tracking import stage_run


def tabular_matrix(df: pd.DataFrame) -> np.ndarray:
    """Encodage simple et fixe des 4 variables cliniques disponibles."""
    age = (df["age_clean"].to_numpy(float) - 47.0) / 17.0
    sex = (df["gender"] == "M").to_numpy(float)
    ap = (df["view"] == "AP").to_numpy(float)
    fu = np.log1p(df["followup"].to_numpy(float))
    return np.column_stack([age, sex, ap, fu])


def tabular_only_auc(df: pd.DataFrame, pathologies: list[str], max_iter: int,
                     seed: int) -> dict:
    tr, va = df[df.split == "train"], df[df.split == "val"]
    xtr, xva = tabular_matrix(tr), tabular_matrix(va)
    out = {}
    for p in pathologies:
        ytr, yva = tr[f"lbl_{p}"].to_numpy(), va[f"lbl_{p}"].to_numpy()
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            out[p] = None
            continue
        clf = LogisticRegression(max_iter=max_iter, class_weight="balanced",
                                 random_state=seed).fit(xtr, ytr)
        out[p] = round(float(roc_auc_score(yva, clf.predict_proba(xva)[:, 1])), 3)
    return out


def run(cfg: Config) -> None:
    pathologies = cfg["preprocess"]["pathologies"]
    src = cfg.processed("dataset.parquet")
    dst = cfg.report("modality_analysis.json")
    with stage_run("analyze", cfg) as rec:
        rec.inputs(src)
        df = pd.read_parquet(src)
        auc = tabular_only_auc(df, pathologies, cfg["analysis"]["max_iter"], cfg["seed"])
        valid = [v for v in auc.values() if v is not None]

        train = df[df.split == "train"]
        pos = {p: int(train[f"lbl_{p}"].sum()) for p in pathologies}
        n = len(train)
        by_view = {p: {v: round(float(x), 4) for v, x in
                       df.groupby("view")[f"lbl_{p}"].mean().items()} for p in pathologies}
        ratio_ap_pa = {p: (round(d["AP"] / d["PA"], 2) if d.get("PA") else None)
                       for p, d in by_view.items() if "AP" in d}
        result = {
            "tabular_only_auc_val": auc,
            "tabular_only_mean_auc": round(float(np.mean(valid)), 3) if valid else None,
            "prevalence_by_view": by_view,
            "prevalence_ratio_ap_over_pa": ratio_ap_pa,
            "view_share_ap": round(float((df.view == "AP").mean()), 3),
            "train_positives": pos,
            "pos_weight": {p: (round((n - k) / k, 1) if k else None) for p, k in pos.items()},
            "images_per_patient": {
                "mean": round(float(df.groupby("patient_id").size().mean()), 2),
                "p95": int(df.groupby("patient_id").size().quantile(0.95)),
                "max": int(df.groupby("patient_id").size().max()),
            },
            "missing_tabular_rate": round(float((df.has_tabular == 0).mean()), 4),
            "missing_image_rate": round(float((df.has_image == 0).mean()), 4),
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        rec.outputs(dst)
        rec.stats(tabular_only_mean_auc=result["tabular_only_mean_auc"],
                  view_share_ap=result["view_share_ap"])
