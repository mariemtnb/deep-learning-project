"""Tests unitaires des fonctions de transformation (sans fichier sur disque)."""
from __future__ import annotations

import pandas as pd

from src.monitoring.drift import compare, psi
from src.preprocessing.clean import add_features, clean
from src.splitting.split import assign_split

P = {"age_min": 1, "age_max": 100, "age_bins": [0, 18, 30, 45, 60, 75, 120],
     "age_labels": ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"],
     "pathologies": ["Effusion", "Mass", "Hernia"]}


def _raw():
    return pd.DataFrame({
        "image": ["a.png", "b.png", "c.png"], "finding_labels": ["No Finding", "Effusion|Mass", "Mass"],
        "followup": [0, 3, 1], "patient_id": [1, 2, 2], "age": [40, 411, 60],
        "gender": ["m ", "F", "F"], "view": ["pa", "AP", "AP"], "width": [2000, 2500, 2500],
        "height": [2000, 2048, 2048], "spacing_x": [0.14] * 3, "spacing_y": [0.14] * 3})


def test_invalid_age_is_flagged_and_replaced_by_valid_median():
    df = clean(_raw(), P)
    assert df["age_invalid"].tolist() == [False, True, False]
    assert df.loc[1, "age_clean"] == 50  # médiane de 40 et 60
    assert df["gender"].tolist() == ["M", "F", "F"]
    assert df["view"].tolist() == ["PA", "AP", "AP"]


def test_labels_are_multi_hot():
    df = add_features(clean(_raw(), P), P)
    assert df[["lbl_Effusion", "lbl_Mass", "lbl_Hernia"]].values.tolist() == [
        [0, 0, 0], [1, 1, 0], [0, 1, 0]]
    assert df["n_findings"].tolist() == [0, 2, 1]
    assert df["is_normal"].tolist() == [1, 0, 0]
    assert df["is_followup"].tolist() == [0, 1, 1]


def _split_input(n_patients=50, per=4):
    rows = [{"image": f"{p:08d}_{k:03d}.png", "patient_id": p + 1}
            for p in range(n_patients) for k in range(per)]
    df = pd.DataFrame(rows)
    official = pd.DataFrame({"image": df["image"],
                             "official_split": ["test" if p < 10 else "trainval"
                                                for p in df["patient_id"] - 1]})
    return df, official


def test_split_keeps_patients_together_and_is_deterministic():
    df, official = _split_input()
    a = assign_split(df, official, 0.2, "patient_id", seed=42)
    b = assign_split(df, official, 0.2, "patient_id", seed=42)
    assert a.equals(b)
    assert (a.groupby("patient_id")["split"].nunique() == 1).all()
    assert set(a["split"]) == {"train", "val", "test"}
    assert a.loc[a.split == "val", "patient_id"].nunique() == 8  # 20 % de 40 patients


def test_psi_is_zero_for_identical_distributions_and_grows_with_shift():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert psi({"AP": 0.4, "PA": 0.6}, {"AP": 0.7, "PA": 0.3}) > 0.25
    ref = {"n_rows": 10, "age": [0.5, 0.5], "gender": {"M": 1.0}, "view": {"PA": 1.0},
           "split": {"train": 1.0}, "prevalence": {"Mass": 0.1}}
    assert compare(ref, ref, 0.1, 0.25)["drift_detected"] is False
