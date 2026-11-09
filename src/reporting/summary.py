"""Rapport de statistiques généré à chaque exécution.

Entrées : processed/dataset.parquet, processed/pneumonia_index.parquet,
          reports/validation_*.json, reports/stages/*.json
Sorties : outputs/summary.json          (statistiques agrégées, lues par le tableau de bord)
          reports/quality_report.html   (rapport qualité HTML autonome)
"""
from __future__ import annotations

import html
import json

import pandas as pd

from src.common.config import Config
from src.common.tracking import stage_run


def build_summary(df: pd.DataFrame, pneu: pd.DataFrame, pathologies: list[str]) -> dict:
    pneu_counts = {f"{s}/{c}": int(n) for (s, c), n in
                   pneu.groupby(["split", "label"]).size().items()}
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "n_images_on_disk": int(df["has_image"].sum()),
        "age": {
            "n_invalid_repaired": int(df["age_invalid"].sum()),
            "mean_clean": round(float(df["age_clean"].mean()), 1),
            "median_clean": int(df["age_clean"].median()),
        },
        "gender_counts": df["gender"].value_counts().to_dict(),
        "view_counts": df["view"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_patients": df.groupby("split")["patient_id"].nunique().to_dict(),
        "normal_vs_abnormal": {"normal": int(df["is_normal"].sum()),
                               "abnormal": int((df["is_normal"] == 0).sum())},
        "multilabel_counts": {str(k): int(v) for k, v in
                              df["n_findings"].value_counts().sort_index().items()},
        "pathology_counts": {p: int(df[f"lbl_{p}"].sum()) for p in pathologies},
        "age_group_counts": df["age_group"].value_counts().to_dict(),
        "pathology_by_gender": {p: df.groupby("gender")[f"lbl_{p}"].sum().to_dict()
                                for p in pathologies},
        "pathology_by_view": {p: {v: round(float(x), 4) for v, x in
                                  df.groupby("view")[f"lbl_{p}"].mean().items()}
                              for p in pathologies},
        "images_per_patient": {
            "mean": round(float(df.groupby("patient_id").size().mean()), 2),
            "max": int(df.groupby("patient_id").size().max()),
        },
        "has_bbox": int(df["has_bbox"].sum()),
        "pneumonia_dataset": pneu_counts,
    }
    return json.loads(json.dumps(summary, default=int))


def _rows(checks: list[dict]) -> str:
    out = []
    for c in checks:
        state = "ok" if c["passed"] else ("fail" if c["level"] == "critical" else "warn")
        label = {"ok": "OK", "fail": "ÉCHEC", "warn": "ALERTE"}[state]
        out.append(
            f"<tr><td><span class='st {state}'>{label}</span></td>"
            f"<td>{html.escape(c['name'])}</td><td>{html.escape(c['level'])}</td>"
            f"<td><code>{html.escape(str(c['value']))}</code></td>"
            f"<td><code>{html.escape(str(c['threshold']))}</code></td>"
            f"<td class='d'>{html.escape(str(c.get('detail', '')))}</td></tr>")
    return "".join(out)


def quality_html(summary: dict, validations: dict, stages: list[dict]) -> str:
    blocks = []
    for title, v in validations.items():
        blocks.append(
            f"<h2>{html.escape(title)}</h2><p>{v['n_checks']} contrôles, "
            f"{v['n_failed_critical']} critique(s) en échec, "
            f"{v['n_failed_warning']} alerte(s).</p>"
            "<table><tr><th></th><th>Contrôle</th><th>Niveau</th><th>Valeur</th>"
            f"<th>Seuil</th><th>Détail</th></tr>{_rows(v['checks'])}</table>")
    stage_rows = "".join(
        f"<tr><td>{html.escape(s['stage'])}</td><td>{html.escape(s.get('status', '?'))}</td>"
        f"<td>{s.get('duration_s', '')} s</td>"
        f"<td class='d'>{html.escape(', '.join(f'{k}={v}' for k, v in s['stats'].items()))}</td></tr>"
        for s in stages)
    return f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<title>Rapport qualité des données</title>
<style>
body{{font:14px/1.5 system-ui,sans-serif;color:#1d2733;max-width:1100px;margin:32px auto;padding:0 16px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border-bottom:1px solid #e3e8ee;padding:6px 8px;text-align:left;vertical-align:top}}
th{{background:#f4f6f8}} code{{font-size:12px}} .d{{color:#5b6b7a;max-width:380px;word-break:break-word}}
.st{{padding:1px 7px;border-radius:3px;font-size:12px;font-weight:600}}
.ok{{background:#e3f4ea;color:#1e7a46}} .fail{{background:#fbe4e4;color:#a12828}}
.warn{{background:#fdf1dc;color:#8a5a00}}
</style></head><body>
<h1>Rapport qualité des données</h1>
<p>Généré le {html.escape(summary['generated_at'].replace('T', ' '))} :
{summary['n_images']:,} images, {summary['n_patients']:,} patients,
{summary['n_images_on_disk']:,} images retrouvées sur disque.</p>
{''.join(blocks)}
<h2>Étapes du pipeline</h2>
<table><tr><th>Étape</th><th>Statut</th><th>Durée</th><th>Statistiques</th></tr>{stage_rows}</table>
</body></html>"""


def run(cfg: Config) -> None:
    pathologies = cfg["preprocess"]["pathologies"]
    src = cfg.processed("dataset.parquet")
    src_pneu = cfg.processed("pneumonia_index.parquet")
    val_files = {"Contrôles sur les données brutes": cfg.report("validation_raw.json"),
                 "Contrôles sur les données préparées": cfg.report("validation_processed.json")}
    out_summary = cfg.path("outputs") / "summary.json"
    out_html = cfg.report("quality_report.html")
    with stage_run("report", cfg) as rec:
        rec.inputs(src, src_pneu, *val_files.values())
        summary = build_summary(pd.read_parquet(src), pd.read_parquet(src_pneu), pathologies)
        out_summary.parent.mkdir(parents=True, exist_ok=True)
        out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        validations = {k: json.loads(p.read_text(encoding="utf-8"))
                       for k, p in val_files.items()}
        order = ["ingest", "index_images", "validate_raw", "preprocess", "align", "split",
                 "validate"]
        stages = [json.loads(p.read_text(encoding="utf-8"))
                  for name in order if (p := cfg.report(f"stages/{name}.json")).exists()]
        out_html.write_text(quality_html(summary, validations, stages), encoding="utf-8")
        rec.outputs(out_summary, out_html)
        rec.stats(images=summary["n_images"], patients=summary["n_patients"])
