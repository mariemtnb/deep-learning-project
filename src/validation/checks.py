"""Contrôles qualité exécutés à chaque passage du pipeline.

Deux niveaux :
- critical : le pipeline s'arrête (code de sortie 1) si le contrôle échoue ;
- warning  : signalé dans le rapport, sans arrêt.

`validate_raw` tourne juste après l'ingestion (données telles qu'elles arrivent),
`validate_processed` juste avant l'export (données prêtes pour le modèle).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd
import pandera.errors as pae

from src.common.config import Config
from src.common.tracking import stage_run

from . import schemas


@dataclass
class CheckResult:
    name: str
    level: str          # "critical" | "warning"
    passed: bool
    value: object
    threshold: object
    detail: str = ""


class QualityGateError(RuntimeError):
    """Levée quand au moins un contrôle critique échoue."""


def _schema(name: str, schema, df: pd.DataFrame) -> CheckResult:
    try:
        schema.validate(df, lazy=True)
        return CheckResult(name, "critical", True, "ok", "schéma respecté")
    except pae.SchemaErrors as err:
        fc = err.failure_cases
        sample = fc[["column", "check", "failure_case"]].head(5).to_dict("records")
        return CheckResult(name, "critical", False, f"{len(fc)} échecs", "0",
                           json.dumps(sample, ensure_ascii=False, default=str))


def _rate(n: int, total: int) -> float:
    return round(n / total, 6) if total else 0.0


# --------------------------------------------------------------------------- #
# Données brutes
# --------------------------------------------------------------------------- #
def raw_checks(cfg: Config, meta: pd.DataFrame, bbox: pd.DataFrame, official: pd.DataFrame,
               index: pd.DataFrame, img_checks: pd.DataFrame) -> list[CheckResult]:
    q, p, im = cfg["quality"], cfg["preprocess"], cfg["images"]
    res = [_schema("schéma table clinique brute", schemas.raw_metadata_schema(), meta),
           _schema("schéma boîtes englobantes", schemas.bbox_schema(), bbox)]

    key_cols = ["image", "patient_id", "age", "gender", "view", "finding_labels"]
    miss = float(meta[key_cols].isna().mean().max())
    res.append(CheckResult("taux de manquants (colonnes clés)", "critical",
                           miss <= q["max_missing_rate"], miss, q["max_missing_rate"]))

    dups = int(meta["image"].duplicated().sum())
    res.append(CheckResult("doublons de nom d'image", "critical",
                           dups <= q["max_duplicate_images"], dups, q["max_duplicate_images"]))

    bad_age = ~pd.to_numeric(meta["age"], errors="coerce").between(p["age_min"], p["age_max"])
    rate = _rate(int(bad_age.sum()), len(meta))
    res.append(CheckResult("âges aberrants (avant réparation)", "critical",
                           rate <= q["max_invalid_age_rate"], rate, q["max_invalid_age_rate"],
                           f"{int(bad_age.sum())} lignes hors [{p['age_min']}, {p['age_max']}]"))

    # Modalité image : dimensions et mode attendus sur l'échantillon contrôlé.
    w, h = im["expected_size"]
    if len(img_checks):
        wrong_size = ~((img_checks.width_px == w) & (img_checks.height_px == h))
        size_rate = _rate(int(wrong_size.sum()), len(img_checks))
        bad_mode = int((~img_checks["mode"].isin(im["allowed_modes"])).sum())
    else:
        size_rate, bad_mode = 1.0, 0
    res.append(CheckResult(f"dimensions des images = {w}x{h}", "critical",
                           size_rate <= q["max_image_size_mismatch_rate"], size_rate,
                           q["max_image_size_mismatch_rate"],
                           f"{len(img_checks)} images ouvertes"))
    res.append(CheckResult("mode des images (niveaux de gris)", "critical", bad_mode == 0,
                           bad_mode, 0, f"modes autorisés : {im['allowed_modes']}"))

    # Cohérence inter-modalités : chaque ligne tabulaire a son image, et inversement.
    meta_imgs, disk_imgs = set(meta["image"]), set(index["image"])
    coverage = _rate(len(meta_imgs & disk_imgs), len(meta_imgs))
    res.append(CheckResult("couverture tabulaire → image", "critical",
                           coverage >= q["min_image_coverage"], coverage, q["min_image_coverage"],
                           f"{len(meta_imgs - disk_imgs)} lignes sans image"))
    orphans = len(disk_imgs - meta_imgs)
    res.append(CheckResult("images sans ligne clinique (orphelines)", "warning", orphans == 0,
                           orphans, 0))

    if q["bbox_must_join"]:
        bad_bbox = len(set(bbox["image"]) - meta_imgs)
        res.append(CheckResult("boîtes englobantes rattachées à une image connue", "critical",
                               bad_bbox == 0, bad_bbox, 0))

    unlisted = len(meta_imgs - set(official["image"]))
    both = int(official["image"].duplicated().sum())
    res.append(CheckResult("listes officielles train_val/test disjointes et complètes",
                           "critical", unlisted == 0 and both == 0,
                           {"absentes": unlisted, "dans_les_deux": both}, 0))
    return res


# --------------------------------------------------------------------------- #
# Données préparées
# --------------------------------------------------------------------------- #
def processed_checks(cfg: Config, df: pd.DataFrame) -> list[CheckResult]:
    q, p = cfg["quality"], cfg["preprocess"]
    res = [_schema("schéma table préparée",
                   schemas.processed_schema(p["pathologies"], p["age_min"], p["age_max"],
                                            p["genders"], p["views"]), df)]

    # Fuite : aucun patient ne doit apparaître dans deux ensembles.
    per_patient = df.groupby(cfg["split"]["group_key"])["split"].nunique()
    leaking = int((per_patient > 1).sum())
    res.append(CheckResult("fuite de patients entre train/val/test", "critical",
                           leaking <= q["max_patients_across_splits"], leaking,
                           q["max_patients_across_splits"]))

    # Masque de modalités cohérent avec les colonnes réelles.
    mask_ok = bool(((df["has_image"] == 1) == df["image_folder"].notna()).all())
    res.append(CheckResult("masque has_image cohérent avec le dossier image", "critical",
                           mask_ok, mask_ok, True))
    complete = _rate(int(((df.has_image == 1) & (df.has_tabular == 1)).sum()), len(df))
    res.append(CheckResult("échantillons avec les deux modalités", "critical",
                           complete >= q["min_image_coverage"], complete,
                           q["min_image_coverage"]))

    # Labels : « No Finding » exclusif, n_findings cohérent.
    lbl = df[[f"lbl_{x}" for x in p["pathologies"]]]
    nf_ok = bool((lbl.sum(axis=1) == df["n_findings"]).all())
    excl = int(((df["is_normal"] == 1) & (df["n_findings"] > 0)).sum())
    res.append(CheckResult("cohérence des labels (n_findings, No Finding exclusif)",
                           "critical", nf_ok and excl == 0,
                           {"n_findings_ok": nf_ok, "normal_avec_label": excl}, 0))

    # Alertes : classes très rares, écart de prévalence val/test.
    prev = lbl.mean()
    rare = prev[prev < q["min_label_prevalence"]]
    res.append(CheckResult("prévalence minimale par pathologie", "warning", rare.empty,
                           {k.removeprefix("lbl_"): round(v, 5) for k, v in rare.items()},
                           q["min_label_prevalence"]))
    by_split = lbl.groupby(df["split"]).mean()
    if {"val", "test"} <= set(by_split.index):
        gap = float((by_split.loc["val"] - by_split.loc["test"]).abs().max())
        res.append(CheckResult("écart de prévalence val / test (max)", "warning",
                               gap <= q["max_val_test_prevalence_gap"], round(gap, 4),
                               q["max_val_test_prevalence_gap"]))
    return res


# --------------------------------------------------------------------------- #
# Exécution
# --------------------------------------------------------------------------- #
def _finish(rec, results: list[CheckResult], out_path) -> None:
    payload = {
        "n_checks": len(results),
        "n_failed_critical": sum(not r.passed and r.level == "critical" for r in results),
        "n_failed_warning": sum(not r.passed and r.level == "warning" for r in results),
        "checks": [asdict(r) for r in results],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8")
    rec.outputs(out_path)
    for r in results:
        mark = "OK  " if r.passed else ("FAIL" if r.level == "critical" else "WARN")
        rec.info(f"{mark} {r.name} (valeur={r.value}, seuil={r.threshold})",
                 check=r.name, passed=r.passed, level=r.level)
    rec.stats(checks=payload["n_checks"], failed_critical=payload["n_failed_critical"],
              failed_warning=payload["n_failed_warning"])
    if payload["n_failed_critical"]:
        names = [r.name for r in results if not r.passed and r.level == "critical"]
        raise QualityGateError(f"contrôle(s) critique(s) en échec : {names}")


def run_raw(cfg: Config) -> None:
    names = ["metadata_raw", "bbox", "official_split", "image_index", "image_checks"]
    srcs = [cfg.interim(f"{n}.parquet") for n in names]
    with stage_run("validate_raw", cfg) as rec:
        rec.inputs(*srcs)
        frames = [pd.read_parquet(s) for s in srcs]
        _finish(rec, raw_checks(cfg, *frames), cfg.report("validation_raw.json"))


def run_processed(cfg: Config) -> None:
    src = cfg.processed("dataset.parquet")
    with stage_run("validate", cfg) as rec:
        rec.inputs(src)
        _finish(rec, processed_checks(cfg, pd.read_parquet(src)),
                cfg.report("validation_processed.json"))
