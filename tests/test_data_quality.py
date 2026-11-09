"""Tests de données : le pipeline passe sur l'échantillon, et chaque contrôle critique
détecte bien le défaut qu'il est censé détecter (on corrompt une copie des données)."""
from __future__ import annotations

import json

import pandas as pd

from src.cli import main
from src.validation.checks import processed_checks, raw_checks


def _failed(results):
    return {r.name for r in results if not r.passed and r.level == "critical"}


def _raw_frames(cfg):
    names = ["metadata_raw", "bbox", "official_split", "image_index", "image_checks"]
    return [pd.read_parquet(cfg.interim(f"{n}.parquet")) for n in names]


def test_pipeline_reports_no_critical_failure(pipeline_done):
    for name in ["validation_raw.json", "validation_processed.json"]:
        report = json.loads(pipeline_done.report(name).read_text(encoding="utf-8"))
        assert report["n_failed_critical"] == 0, report


def test_processed_table_has_every_modality(processed):
    assert processed["has_image"].all() and processed["has_tabular"].all()
    assert processed["image_folder"].notna().all()


def test_no_patient_in_two_splits(processed):
    assert (processed.groupby("patient_id")["split"].nunique() == 1).all()


def test_leakage_is_caught(pipeline_done, processed):
    bad = processed.copy()
    counts = bad[bad.split == "train"].groupby("patient_id").size()
    victim = counts[counts >= 2].index[0]  # un patient avec au moins deux images
    bad.loc[bad.index[bad.patient_id == victim][0], "split"] = "test"
    assert "fuite de patients entre train/val/test" in _failed(processed_checks(pipeline_done, bad))


def test_label_inconsistency_is_caught(pipeline_done, processed):
    bad = processed.copy()
    row = bad.index[bad.is_normal == 1][0]
    bad.loc[row, "lbl_Mass"] = 1
    assert any("cohérence des labels" in n for n in _failed(processed_checks(pipeline_done, bad)))


def test_missing_image_is_caught(pipeline_done):
    meta, bbox, official, index, img = _raw_frames(pipeline_done)
    index = index.iloc[1:]  # une image disparaît du disque
    assert "couverture tabulaire → image" in _failed(
        raw_checks(pipeline_done, meta, bbox, official, index, img))


def test_duplicates_and_bad_values_are_caught(pipeline_done):
    meta, bbox, official, index, img = _raw_frames(pipeline_done)
    meta = pd.concat([meta, meta.head(1)], ignore_index=True)
    meta.loc[0, "gender"] = "X"
    failed = _failed(raw_checks(pipeline_done, meta, bbox, official, index, img))
    assert "doublons de nom d'image" in failed
    assert "schéma table clinique brute" in failed


def test_wrong_image_size_is_caught(pipeline_done):
    meta, bbox, official, index, img = _raw_frames(pipeline_done)
    img = img.assign(width_px=64)
    assert any(n.startswith("dimensions des images") for n in _failed(
        raw_checks(pipeline_done, meta, bbox, official, index, img)))


def test_rerun_is_reproducible(pipeline_done):
    """Même code + même config + mêmes données = même table, octet pour octet."""
    path = pipeline_done.processed("dataset.parquet")
    before = pd.read_parquet(path)
    assert main(["split", "--config", "configs/sample.yaml"]) == 0
    pd.testing.assert_frame_equal(before, pd.read_parquet(path))
