"""
Build the project dashboard / report from outputs/summary.json + outputs/examples.json.

Produces a fully self-contained static HTML page: every chart is inline SVG, so the
page needs no internet, no CDN and no JavaScript, and renders identically everywhere.

Pipeline stage `dashboard` (see dvc.yaml)  ->  outputs/dashboard.html
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The pipeline CLI sets DL_OUTPUTS_DIR from the config (paths.outputs).
OUT = Path(os.environ.get("DL_OUTPUTS_DIR", ROOT / "outputs"))
S = json.loads((OUT / "summary.json").read_text())
EX = json.loads((OUT / "examples.json").read_text(encoding="utf-8"))

INK = "#15202b"; MUT = "#5b6b7a"; FAINT = "#8a97a3"
FILL = "#2f88a6"; NEU = "#c7d0d8"; ACC = "#1f6f8b"; LINE = "#e6eaef"


def esc(x) -> str:
    return html.escape(str(x))


def fmt(n) -> str:
    return f"{int(n):,}"


# --------------------------------------------------------------------------- #
# SVG chart helpers (pure, no dependencies)
# --------------------------------------------------------------------------- #
def svg_open(w: int, h: int) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" '
            f'font-family="ui-sans-serif,system-ui,sans-serif" '
            f'role="img" preserveAspectRatio="xMinYMin meet">')


def hbars(pairs, label_w=118, bar_h=22, gap=9, val_w=58, color=FILL, top=6):
    """Horizontal bar chart. pairs = [(label, value), ...]."""
    w = 680
    plot_w = w - label_w - val_w
    mx = max(v for _, v in pairs) or 1
    h = top + len(pairs) * (bar_h + gap)
    out = [svg_open(w, h)]
    y = top
    for lbl, v in pairs:
        bw = plot_w * v / mx
        out.append(f'<text x="{label_w-8}" y="{y+bar_h*0.72}" text-anchor="end" '
                   f'font-size="12.5" fill="{INK}">{esc(lbl)}</text>')
        out.append(f'<rect x="{label_w}" y="{y}" width="{bw:.1f}" height="{bar_h}" '
                   f'rx="2" fill="{color}"/>')
        out.append(f'<text x="{label_w+bw+7:.1f}" y="{y+bar_h*0.72}" font-size="12" '
                   f'fill="{MUT}" font-variant-numeric="tabular-nums">{fmt(v)}</text>')
        y += bar_h + gap
    out.append("</svg>")
    return "".join(out)


def vbars(pairs, color=FILL, bar_w=None):
    """Vertical bar chart. pairs = [(label, value), ...]."""
    w = 680; h = 240; base = h - 34; top = 18
    n = len(pairs)
    slot = w / n
    bw = bar_w or min(64, slot * 0.6)
    mx = max(v for _, v in pairs) or 1
    out = [svg_open(w, h)]
    for i, (lbl, v) in enumerate(pairs):
        cx = slot * (i + 0.5)
        bh = (base - top) * v / mx
        out.append(f'<rect x="{cx-bw/2:.1f}" y="{base-bh:.1f}" width="{bw:.1f}" '
                   f'height="{bh:.1f}" rx="2" fill="{color}"/>')
        out.append(f'<text x="{cx:.1f}" y="{base-bh-6:.1f}" text-anchor="middle" '
                   f'font-size="11.5" fill="{MUT}" font-variant-numeric="tabular-nums">{fmt(v)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base+18:.1f}" text-anchor="middle" '
                   f'font-size="12" fill="{INK}">{esc(lbl)}</text>')
    out.append(f'<line x1="0" y1="{base}" x2="{w}" y2="{base}" stroke="{LINE}"/>')
    out.append("</svg>")
    return "".join(out)


def stacked_single(segments):
    """One horizontal bar split into segments = [(label, value, color), ...]."""
    w = 680; h = 74; bh = 40; top = 6
    total = sum(v for _, v, _ in segments) or 1
    out = [svg_open(w, h)]
    x = 0.0
    for lbl, v, col in segments:
        sw = w * v / total
        out.append(f'<rect x="{x:.1f}" y="{top}" width="{sw:.1f}" height="{bh}" fill="{col}"/>')
        if sw > 90:
            pct = 100 * v / total
            out.append(f'<text x="{x+sw/2:.1f}" y="{top+bh*0.62:.1f}" text-anchor="middle" '
                       f'font-size="13" fill="#fff" font-weight="600">{pct:.0f}%</text>')
        x += sw
    # legend
    lx = 0
    for lbl, v, col in segments:
        out.append(f'<rect x="{lx}" y="{top+bh+12}" width="11" height="11" fill="{col}"/>')
        out.append(f'<text x="{lx+16}" y="{top+bh+22}" font-size="12.5" fill="{INK}">'
                   f'{esc(lbl)} ({fmt(v)})</text>')
        lx += 20 + len(f"{lbl} ({fmt(v)})") * 7.2
    out.append("</svg>")
    return "".join(out)


def stacked_groups(groups, series):
    """Vertical stacked bars. groups=[labels]; series=[(name,[vals],color)]."""
    w = 680; h = 250; base = h - 44; top = 18
    n = len(groups); slot = w / n; bw = min(80, slot * 0.5)
    totals = [sum(s[1][i] for s in series) for i in range(n)]
    mx = max(totals) or 1
    out = [svg_open(w, h)]
    for i, g in enumerate(groups):
        cx = slot * (i + 0.5); y = base
        for name, vals, col in series:
            bh = (base - top) * vals[i] / mx
            out.append(f'<rect x="{cx-bw/2:.1f}" y="{y-bh:.1f}" width="{bw:.1f}" '
                       f'height="{bh:.1f}" fill="{col}"/>')
            y -= bh
        out.append(f'<text x="{cx:.1f}" y="{base-((base-top)*totals[i]/mx)-6:.1f}" '
                   f'text-anchor="middle" font-size="11.5" fill="{MUT}" '
                   f'font-variant-numeric="tabular-nums">{fmt(totals[i])}</text>')
        out.append(f'<text x="{cx:.1f}" y="{base+18:.1f}" text-anchor="middle" '
                   f'font-size="12" fill="{INK}">{esc(g)}</text>')
    out.append(f'<line x1="0" y1="{base}" x2="{w}" y2="{base}" stroke="{LINE}"/>')
    lx = 0
    for name, vals, col in series:
        out.append(f'<rect x="{lx}" y="{h-12}" width="11" height="11" fill="{col}"/>')
        out.append(f'<text x="{lx+16}" y="{h-2}" font-size="12.5" fill="{INK}">{esc(name)}</text>')
        lx += 30 + len(name) * 7.5
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Charts from the summary
# --------------------------------------------------------------------------- #
path_pairs = sorted(S["pathology_counts"].items(), key=lambda x: -x[1])
path_pairs = [(k.replace("_", " "), v) for k, v in path_pairs]
CH_PATH = hbars(path_pairs, label_w=118, bar_h=20, gap=8)

CH_NORM = stacked_single([
    ("Normal", S["normal_vs_abnormal"]["normal"], NEU),
    ("Abnormal", S["normal_vs_abnormal"]["abnormal"], FILL)])

multi = [(k, v) for k, v in sorted(S["multilabel_counts"].items(), key=lambda x: int(x[0]))]
CH_MULTI = vbars(multi)

split_order = [k for k in ["train", "val", "test"] if k in S["split_counts"]]
CH_SPLIT = vbars([(k, S["split_counts"][k]) for k in split_order])

age_order = [k for k in ["0-18", "19-30", "31-45", "46-60", "61-75", "76+"]
             if k in S["age_group_counts"]]
CH_AGE = vbars([(k, S["age_group_counts"][k]) for k in age_order])

CH_GENDER = hbars([("Male" if k == "M" else "Female", v)
                   for k, v in S["gender_counts"].items()],
                  label_w=70, bar_h=28, gap=12,
                  color=FILL)
CH_VIEW = hbars([(k, v) for k, v in S["view_counts"].items()],
                label_w=70, bar_h=28, gap=12, color=FILL)

sp = ["train", "val", "test"]
CH_PNEU = stacked_groups(sp, [
    ("Normal", [S["pneumonia_dataset"].get(f"{s}/NORMAL", 0) for s in sp], NEU),
    ("Pneumonia", [S["pneumonia_dataset"].get(f"{s}/PNEUMONIA", 0) for s in sp], FILL)])

# --------------------------------------------------------------------------- #
# Static report content (from the project study document)
# --------------------------------------------------------------------------- #
OBJECTIVES = [
    "Classify chest pathologies (14 NIH labels, multi-label; or binary pneumonia / normal) from an X-ray plus clinical data.",
    "Compare three models: image-only, tabular-only, and multimodal (late fusion, then attention fusion).",
    "Build a reproducible, anonymised data pipeline with no data leakage between patients.",
    "Provide explainability (Grad-CAM for the image, SHAP for the clinical features) for medical acceptability.",
]
SOURCES = [
    ["NIH Chest X-ray14 (Kaggle)", "112,120 X-rays, 30,805 patients, 14 labels + metadata (age, sex, PA/AP view, follow-up #)", "Open (Kaggle)", "Primary dataset: image + tabular", "Used"],
    ["Chest X-Ray Pneumonia (Kaggle)", "5,863 pediatric images, 2 classes", "Open", "Quick prototype / external check", "Used"],
    ["MIMIC-CXR (PhysioNet)", "377,110 images, radiology reports, link to MIMIC-IV (labs, history)", "PhysioNet licence + CITI training", "Extension: rich clinical data", "Planned"],
    ["MIMIC-IV / eICU (PhysioNet)", "Labs, vitals, ICD diagnoses", "Conditional access", "Enrich clinical features", "Planned"],
]
SCHEMA = [
    ["image", "string", "File name, the join key to the image on disk."],
    ["patient_id", "int", "Patient identifier (multiple images per patient)."],
    ["age_clean", "int", "Age after repairing impossible values."],
    ["gender", "cat", "M / F."],
    ["view", "cat", "Projection: PA or AP."],
    ["followup", "int", "Follow-up number for that patient."],
    ["width, height", "int", "Original pixel dimensions."],
    ["finding_labels", "string", "Pipe-separated findings, e.g. Cardiomegaly|Effusion."],
    ["lbl_* (x14)", "0/1", "One binary column per pathology (multi-hot)."],
    ["n_findings", "int", "Count of positive labels on the image."],
    ["split", "cat", "train / val / test, assigned by patient."],
    ["has_bbox", "0/1", "Whether a bounding box annotation exists."],
]
PIPELINE = [
    ["Ingestion", "Read Data_Entry_2017.csv and the official train_val / test lists; index all 112,120 images across 12 folders.", "Done"],
    ["Schema validation", "Check expected columns and types; validate ranges (age 0 to 100, sex M/F, view PA/AP).", "Done"],
    ["Alignment image to folder", "Join by file name, verify each image exists on disk, report orphans.", "Done"],
    ["Cleaning", "Repair impossible ages (>100), normalise categoricals.", "Done"],
    ["Feature engineering", "Multi-hot encode the 14 labels, age bands, follow-up flag, aspect ratio.", "Done"],
    ["Patient-level split", "Group by patient so no patient spans two splits; verified zero overlap.", "Done"],
    ["Export", "Parquet table versioned with DVC, JSON summary and HTML quality report.", "Done"],
    ["Quality gates", "pandera schemas + leakage / cross-modal checks; the pipeline stops on a critical failure.", "Done"],
    ["Automation", "dvc repro rebuilds everything from the raw sources; GitHub Actions runs it on a sample.", "Done"],
    ["Anonymisation", "Salted SHA-256 pseudonymisation of patient IDs, PHI tag stripping for DICOM.", "Planned"],
    ["Missing values", "Train-only imputation + binary is_missing indicators (once MIMIC labs are added).", "Planned"],
    ["Image preprocessing", "Grayscale, CLAHE, resize 224x224, ImageNet normalisation, train-only augmentation.", "Planned"],
    ["Serialise for training", "scikit-learn preprocessor (joblib), class weights, multimodal PyTorch DataLoader.", "Planned"],
]
ARCH = [
    ["Image branch", "X-ray 224x224 into a pretrained backbone (DenseNet-121 / EfficientNet-B0 / ViT-B/16) into a 256-d vector."],
    ["Tabular branch", "Normalised clinical features + missingness flags into an MLP (64-32) into a 32-d vector."],
    ["Fusion head", "Concatenation (late) or cross-attention, then a dense layer and sigmoids for multi-label output."],
]
FUSION = [
    ["Early fusion", "Concatenate raw inputs", "Simple", "Incompatible image/tabular scales"],
    ["Late fusion", "Concat embeddings, then a classifier", "Robust, modular, the baseline", "Few cross-modal interactions"],
    ["Decision fusion", "Weighted average of probabilities", "Very interpretable", "No interaction at all"],
    ["Attention fusion", "Cross-attention / FiLM: clinical data modulates visual features", "Rich interactions, best performance", "More complex, overfitting risk"],
]
RISKS = [
    ["Leakage between patients", "Overstated performance", "Group split by patient"],
    ["Class imbalance", "Rare classes ignored", "Focal loss, class weights, AUC-PR"],
    ["Noisy NLP labels (~10%)", "Performance ceiling", "Label smoothing, validate on an annotated subset"],
    ["Demographic bias", "Unfair model", "Evaluate per subgroup (sex, age band)"],
    ["Shortcut learning (AP view, markers)", "Poor generalisation", "Grad-CAM, external validation"],
]
STACK = [
    ["Data", "pandas, numpy, pyarrow"], ["EDA / viz", "matplotlib, seaborn, plotly"],
    ["Missing values", "scikit-learn imputers, missingno"],
    ["Tabular preprocessing", "ColumnTransformer, StandardScaler, OneHotEncoder"],
    ["Image", "OpenCV, Pillow, scikit-image, pydicom"],
    ["Augmentation", "albumentations, torchvision"],
    ["Leak-free split", "GroupShuffleSplit, iterative-stratification"],
    ["Anonymisation", "hashlib (SHA-256 + salt), pydicom, presidio"],
    ["Validation", "pandera / Great Expectations"],
    ["Deep learning", "PyTorch, timm, MONAI, TorchXRayVision"],
    ["Training", "PyTorch Lightning, torchmetrics"],
    ["Tracking / tuning", "MLflow or W&B, Optuna"],
    ["Explainability", "pytorch-grad-cam, SHAP, Captum"],
    ["Serving", "FastAPI, Streamlit / Gradio, ONNX"],
]


# --------------------------------------------------------------------------- #
# HTML builders
# --------------------------------------------------------------------------- #
def chip(status: str) -> str:
    cls = {"Used": "", "Done": "", "Planned": "plan", "Partial": "part"}.get(status, "")
    return f'<span class="chip {cls}">{esc(status)}</span>'


def table(head, rows, cell_cls=None) -> str:
    cell_cls = cell_cls or {}
    thead = "".join(f"<th>{esc(h)}</th>" for h in head)
    body = []
    for r in rows:
        tds = []
        for i, c in enumerate(r):
            cls = cell_cls.get(i, "")
            tds.append(f'<td class="{cls}">{c}</td>' if cls else f"<td>{c}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"<table><tr>{thead}</tr>{''.join(body)}</table>"


def shot(e) -> str:
    meta = " &middot; ".join(f"{esc(k)}: {esc(v)}" for k, v in e["meta"].items())
    return (f'<figure class="shot"><img src="{e["img"]}" alt="{esc(e["caption"])}">'
            f'<figcaption class="cap"><div class="t">{esc(e["caption"])}</div>'
            f'<div class="m">{meta}</div></figcaption></figure>')


facts = [
    (fmt(S["n_images"]), "X-ray images"), (fmt(S["n_patients"]), "patients"),
    ("14", "pathology classes"), ("2", "modalities"),
    (fmt(S["has_bbox"]), "box-annotated"), ("0", "patient leakage"),
]
FACTS = "".join(f'<div class="fact"><div class="n">{v}</div>'
                f'<div class="k">{esc(k)}</div></div>' for v, k in facts)
OBJ = "".join(f"<li>{esc(o)}</li>" for o in OBJECTIVES)

sr = EX["sample_row"]
SAMPLE = "<br>".join(f'<span class="kk">{esc(k)}</span>: {esc(v)}' for k, v in sr.items())

T_SOURCES = table(["Source", "Contents", "Access", "Role", ""],
                  [[esc(a), esc(b), esc(c), esc(d), chip(e)] for a, b, c, d, e in SOURCES],
                  {0: "k", 4: "s"})
T_SCHEMA = table(["Column", "Type", "Meaning"],
                 [[f"<code>{esc(a)}</code>", esc(b), esc(c)] for a, b, c in SCHEMA], {0: "k"})
T_PIPE = table(["Stage", "What happens", "Status"],
               [[esc(a), esc(b), chip(c)] for a, b, c in PIPELINE], {0: "step", 2: "s"})
T_ARCH = table(["Branch", "Design"], [[esc(a), esc(b)] for a, b in ARCH], {0: "k"})
T_FUSION = table(["Strategy", "Principle", "Pros", "Cons"],
                 [[esc(a), esc(b), esc(c), esc(d)] for a, b, c, d in FUSION], {0: "k"})
T_RISK = table(["Risk", "Impact", "Safeguard"],
               [[esc(a), esc(b), esc(c)] for a, b, c in RISKS], {0: "k"})
T_STACK = table(["Stage", "Libraries"],
                [[esc(a), f"<code>{esc(b)}</code>"] for a, b in STACK], {0: "k"})

NIH_GALLERY = "".join(shot(e) for e in EX["nih"])
BBOX = shot(EX["bbox"]) if EX.get("bbox") else ""
PNEU_GALLERY = "".join(shot(e) for e in EX["pneumonia"])

GENAT = esc(S["generated_at"].replace("T", " "))


def chart_block(title, svg):
    return (f'<h3 class="ch">{esc(title)}</h3>'
            f'<div class="chart">{svg}</div>')


DOC = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Assisted medical diagnosis: project report</title>
<style>
  :root{{--ink:{INK};--muted:{MUT};--faint:{FAINT};--line:{LINE};--accent:{ACC};
    --warn:#9a6a1a;--warnbg:#f7efe0}}
  *{{box-sizing:border-box}}
  body{{margin:0;background:#fff;color:var(--ink);font:15.5px/1.6 Georgia,ui-serif,serif}}
  a{{color:var(--accent)}}
  .sans{{font-family:ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}}
  header,nav,main,footer{{max-width:980px;margin:0 auto;padding-left:24px;padding-right:24px}}
  header{{padding-top:48px;padding-bottom:8px}}
  .eyebrow{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;
    letter-spacing:.14em;text-transform:uppercase;color:var(--accent);margin:0 0 10px}}
  h1{{font-size:32px;line-height:1.12;margin:0 0 12px;font-weight:600}}
  .lede{{color:var(--muted);max-width:64ch;margin:0}}
  .disclaimer{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12.5px;
    color:var(--warn);background:var(--warnbg);border:1px solid #ecdcbf;border-radius:4px;
    padding:9px 13px;margin:16px 0 0;max-width:64ch}}
  nav{{padding:14px 24px;font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px;
    border-bottom:1px solid var(--line)}}
  nav a{{color:var(--muted);text-decoration:none;margin-right:16px;white-space:nowrap}}
  nav a:hover{{color:var(--accent)}}
  main{{padding-bottom:80px}}
  hr{{border:0;border-top:1px solid var(--line);margin:34px 0}}
  section{{margin:40px 0;scroll-margin-top:12px}}
  .snum{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:12px;letter-spacing:.1em;color:var(--faint)}}
  h2{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:19px;margin:2px 0 6px;font-weight:700}}
  h3.ch{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:13.5px;font-weight:600;
    margin:20px 0 6px;color:var(--ink)}}
  .note{{color:var(--muted);font-size:14.5px;margin:0 0 14px;max-width:72ch}}
  ul.obj{{margin:0;padding-left:20px}} ul.obj li{{margin:6px 0;max-width:72ch}}
  .facts{{display:grid;grid-template-columns:repeat(6,1fr);border:1px solid var(--line);
    border-radius:4px;overflow:hidden;margin:20px 0 6px}}
  .fact{{padding:14px;border-right:1px solid var(--line)}}
  .fact:last-child{{border-right:0}}
  .fact .n{{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}}
  .fact .k{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:10.5px;letter-spacing:.05em;
    text-transform:uppercase;color:var(--faint);margin-top:3px}}
  @media(max-width:820px){{.facts{{grid-template-columns:repeat(3,1fr)}}
    .fact{{border-bottom:1px solid var(--line)}}}}
  @media(max-width:520px){{.facts{{grid-template-columns:repeat(2,1fr)}}}}
  table{{width:100%;border-collapse:collapse;font-size:14px;margin:4px 0}}
  td,th{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
  th{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11px;letter-spacing:.05em;
    text-transform:uppercase;color:var(--faint);font-weight:600}}
  td.k,td.step{{font-family:ui-sans-serif,system-ui,sans-serif;font-weight:600;white-space:nowrap}}
  td.s{{white-space:nowrap;width:1%}}
  code{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;background:#f3f5f7;
    padding:1px 5px;border-radius:3px}}
  .chip{{font-family:ui-sans-serif,system-ui,sans-serif;font-size:11px;font-weight:600;
    padding:2px 9px;border-radius:999px;white-space:nowrap;background:#eef4f6;color:var(--accent)}}
  .chip.plan{{background:#f1f3f5;color:#6b7784}} .chip.part{{background:#f7efe0;color:#9a6a1a}}
  .chart{{border:1px solid var(--line);border-radius:4px;padding:12px 14px;background:#fff}}
  .row{{display:grid;grid-template-columns:1fr 1fr;gap:34px;align-items:start}}
  @media(max-width:760px){{.row{{grid-template-columns:1fr;gap:24px}}}}
  .gallery{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;align-items:start;margin-top:6px}}
  @media(max-width:760px){{.gallery{{grid-template-columns:repeat(2,1fr)}}}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:14px;align-items:start;max-width:520px}}
  .shot{{margin:0;border:1px solid var(--line);border-radius:4px;overflow:hidden;background:#000}}
  .shot img{{display:block;width:100%;height:auto}}
  .cap{{background:#fff;padding:8px 10px}}
  .cap .t{{font-family:ui-sans-serif,system-ui,sans-serif;font-weight:600;font-size:13px}}
  .cap .m{{color:var(--faint);font-size:11.5px;font-family:ui-sans-serif,system-ui,sans-serif;
    margin-top:2px;line-height:1.45;word-break:break-word}}
  .sample{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;background:#f7f9fa;
    border:1px solid var(--line);border-radius:4px;padding:14px 16px;overflow:auto}}
  .sample .kk{{color:var(--accent)}}
  footer{{color:var(--faint);font-family:ui-sans-serif,system-ui,sans-serif;font-size:12.5px;
    padding-bottom:56px;line-height:1.6}}
</style>
</head>
<body>
<header>
  <p class="eyebrow">Project 1 &middot; Assisted medical diagnosis</p>
  <h1>Multimodal classification of chest X-rays with clinical data</h1>
  <p class="lede">A radiologist reads the image, but also the patient's age, sex and
  history. Image-only deep models ignore that context. This project builds a decision
  support system that combines both, and it starts with a clean, leak-free data
  pipeline. This page documents the task, the data, and every step taken so far.</p>
  <p class="disclaimer">Educational and research tool. Not a certified medical device
  and not a substitute for clinical diagnosis.</p>
</header>

<nav>
  <a href="#task">Task</a><a href="#data">Data &amp; sources</a>
  <a href="#what">What the data is</a><a href="#examples">Examples</a>
  <a href="#viz">Visualisations</a><a href="#pipeline">Pipeline</a>
  <a href="#model">Model plan</a><a href="#risks">Risks</a>
</nav>

<main>
  <div class="facts">{FACTS}</div>

  <section id="task"><div class="snum">01</div><h2>The task</h2>
    <p class="note">Classify thoracic pathologies from a chest X-ray together with
    structured clinical data, and show that fusing the two beats either one alone.</p>
    <ul class="obj">{OBJ}</ul></section>
  <hr>

  <section id="data"><div class="snum">02</div><h2>Data and how it was collected</h2>
    <p class="note">The images are public, de-identified chest radiographs released by
    the US National Institutes of Health; the 14 labels were mined from the original
    radiology reports with NLP (hence ~10% label noise). Metadata is aligned to each
    image by file name in <code>Data_Entry_2017.csv</code>. A separate pediatric
    pneumonia set is used for a quick binary prototype. MIMIC is the planned clinical
    extension.</p>{T_SOURCES}</section>
  <hr>

  <section id="what"><div class="snum">03</div><h2>What this data is</h2>
    <p class="note">After processing, each row is one X-ray. The columns below are the
    processed schema; one real row from the table follows.</p>
    <div class="row">{T_SCHEMA}
      <div><div class="note" style="margin-bottom:8px"><strong>One sample row</strong>
        (<code>processed_metadata.csv</code>)</div>
        <div class="sample">{SAMPLE}</div></div></div></section>
  <hr>

  <section id="examples"><div class="snum">04</div><h2>What it looks like</h2>
    <p class="note">Real images from the dataset, downscaled. Each carries the clinical
    metadata that the tabular branch will use.</p>
    <div class="gallery">{NIH_GALLERY}</div>
    <div class="row" style="margin-top:26px">
      <div><div class="note" style="margin-bottom:8px"><strong>Localised finding.</strong>
        880 images carry a bounding box (orange) marking where the pathology is.</div>
        <div style="max-width:360px">{BBOX}</div></div>
      <div><div class="note" style="margin-bottom:8px"><strong>Pneumonia prototype set.</strong>
        A separate binary (normal vs pneumonia) pediatric dataset.</div>
        <div class="two">{PNEU_GALLERY}</div></div></div></section>
  <hr>

  <section id="viz"><div class="snum">05</div><h2>Exploratory visualisation</h2>
    <p class="note">The shape of the data that drives every modelling choice: heavy
    class imbalance, frequent co-occurrence, and a demographic skew. Charts are inline
    SVG, so this page needs no internet or scripts.</p>
    {chart_block("Pathology prevalence (images positive per finding)", CH_PATH)}
    <div class="row">
      <div>{chart_block("Normal vs abnormal", CH_NORM)}</div>
      <div>{chart_block("Findings co-occurring per image", CH_MULTI)}</div></div>
    <div class="row">
      <div>{chart_block("Patient-level split", CH_SPLIT)}</div>
      <div>{chart_block("Age at acquisition", CH_AGE)}</div></div>
    <div class="row">
      <div>{chart_block("Gender", CH_GENDER)}</div>
      <div>{chart_block("View position", CH_VIEW)}</div></div>
    {chart_block("Pneumonia set by split", CH_PNEU)}</section>
  <hr>

  <section id="pipeline"><div class="snum">06</div><h2>The data pipeline (DataOps)</h2>
    <p class="note">Idempotent stages, each producing a versioned artifact. Status
    reflects this repository: the metadata-preparation stages are implemented;
    anonymisation, imputation and image preprocessing are specified for the next phase
    (essential once MIMIC clinical data and DICOM images are added).</p>{T_PIPE}</section>
  <hr>

  <section id="model"><div class="snum">07</div><h2>Model plan (beyond the pipeline)</h2>
    <p class="note">Three branches, compared in stages: image-only and tabular-only
    baselines first, then two fusion strategies.</p>{T_ARCH}
    <h3 class="ch">Fusion strategies compared</h3>{T_FUSION}
    <p class="note" style="margin-top:14px"><strong>Loss:</strong> class-weighted BCE or
    focal loss (imbalance). <strong>Metrics:</strong> per-class and mean AUC-ROC (the
    NIH reference), F1, sensitivity, specificity, AUC-PR, plus calibration (reliability
    curve, ECE). <strong>Explainability:</strong> Grad-CAM over the image, SHAP over the
    clinical features.</p>
    <h3 class="ch">Tooling</h3>{T_STACK}</section>
  <hr>

  <section id="risks"><div class="snum">08</div><h2>Risks and safeguards</h2>{T_RISK}</section>
</main>

<footer>Generated {GENAT} from <code>outputs/summary.json</code> and
  <code>outputs/examples.json</code> by <code>src/reporting/dashboard.py</code>.
  NIH labels are NLP-extracted from radiology reports (~10% estimated noise).
  Images: NIH ChestX-ray14 (Wang et al., CVPR 2017) and the Kaggle pediatric pneumonia
  set, both public and de-identified.</footer>
</body>
</html>
"""

out = OUT / "dashboard.html"
out.write_text(DOC, encoding="utf-8")
print(f"[dashboard] wrote {out} ({out.stat().st_size/1024:.0f} KB, static SVG, no CDN)")
