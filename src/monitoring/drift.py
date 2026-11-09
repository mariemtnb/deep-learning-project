"""Détection de dérive entre deux versions des données préparées.

Entrées : processed/dataset.parquet + un profil de référence (reports/reference_profile.json)
Sortie  : reports/drift.json

Le profil résume la distribution de l'âge (déciles fixes), du sexe, de la vue, des
ensembles et de la prévalence de chaque pathologie. On compare avec l'indice de
stabilité de population (PSI) : < 0,10 stable, 0,10 à 0,25 à surveiller, > 0,25 dérive.

Usage manuel :
    python -m src.cli drift                      # compare à la référence
    python -m src.cli drift --set-reference      # la version courante devient la référence
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import ROOT, Config
from src.common.tracking import stage_run

AGE_EDGES = [0, 18, 25, 32, 39, 46, 53, 60, 67, 74, 120]


def profile(df: pd.DataFrame, pathologies: list[str]) -> dict:
    age_hist = np.histogram(df["age_clean"], bins=AGE_EDGES)[0]
    return {
        "n_rows": int(len(df)),
        "age": (age_hist / age_hist.sum()).round(6).tolist(),
        "gender": df["gender"].value_counts(normalize=True).round(6).to_dict(),
        "view": df["view"].value_counts(normalize=True).round(6).to_dict(),
        "split": df["split"].value_counts(normalize=True).round(6).to_dict(),
        "prevalence": {p: round(float(df[f"lbl_{p}"].mean()), 6) for p in pathologies},
    }


def psi(ref: list[float] | dict, cur: list[float] | dict, eps: float = 1e-4) -> float:
    if isinstance(ref, dict):
        keys = sorted(set(ref) | set(cur))
        ref, cur = [ref.get(k, 0) for k in keys], [cur.get(k, 0) for k in keys]
    r = np.clip(np.asarray(ref, float), eps, None)
    c = np.clip(np.asarray(cur, float), eps, None)
    return round(float(np.sum((c - r) * np.log(c / r))), 4)


def compare(ref: dict, cur: dict, warn: float, alert: float) -> dict:
    scores = {k: psi(ref[k], cur[k]) for k in ["age", "gender", "view", "split"]}
    prev_shift = {p: round(cur["prevalence"][p] - ref["prevalence"].get(p, 0), 5)
                  for p in cur["prevalence"]}

    def level(x: float) -> str:
        return "alert" if x > alert else ("warn" if x > warn else "stable")

    return {
        "rows": {"reference": ref["n_rows"], "current": cur["n_rows"]},
        "psi": scores,
        "levels": {k: level(v) for k, v in scores.items()},
        "prevalence_shift": prev_shift,
        "max_abs_prevalence_shift": max(abs(v) for v in prev_shift.values()),
        "drift_detected": any(v > alert for v in scores.values()),
    }


def run(cfg: Config, set_reference: bool = False) -> None:
    d = cfg["drift"]
    pathologies = cfg["preprocess"]["pathologies"]
    src = cfg.processed("dataset.parquet")
    ref_path = ROOT / d["reference"]
    dst = cfg.report("drift.json")
    with stage_run("drift", cfg) as rec:
        rec.inputs(src)
        cur = profile(pd.read_parquet(src), pathologies)
        if set_reference or not ref_path.exists():
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
            rec.info(f"profil de référence écrit : {ref_path.name}")
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        result = compare(ref, cur, d["psi_warn"], d["psi_alert"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rec.outputs(dst)
        rec.stats(drift_detected=result["drift_detected"], **result["psi"])
