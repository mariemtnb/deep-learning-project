"""
Export example X-ray thumbnails + a sample metadata row to outputs/examples.json,
so the dashboard can show what the raw data actually looks like.

Pipeline stage `examples` (see dvc.yaml).
"""
from __future__ import annotations

import base64
import io
import json

import pandas as pd
from PIL import Image, ImageDraw

from src.common.config import Config
from src.common.tracking import stage_run

THUMB = 360
NIH = None  # set by run() from the config


def to_b64(img: Image.Image, color: bool = False) -> str:
    img = img.convert("RGB" if color else "L")
    img.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def nih_example(fname: str, folder: str, caption: str, meta: dict, bbox=None) -> dict:
    path = NIH / folder / "images" / fname
    if not path.exists():
        return {"caption": caption, "meta": meta, "img": ""}
    # Draw the box on a grayscale-then-RGB copy so the orange outline stays visible.
    base = Image.open(path).convert("L").convert("RGB")
    if bbox is not None:
        x, y, w, h = bbox
        draw = ImageDraw.Draw(base)
        draw.rectangle([x, y, x + w, y + h], outline=(240, 100, 40), width=14)
    return {"caption": caption, "meta": meta, "img": to_b64(base, color=bbox is not None)}


def build(cfg: Config) -> dict:
    global NIH, THUMB
    NIH = cfg.path("nih_images")
    pneu_root = cfg.path("pneumonia_images")
    THUMB = cfg["examples"]["thumb_px"]
    df = pd.read_parquet(cfg.processed("dataset.parquet"))
    bbox = pd.read_parquet(cfg.interim("bbox.parquet"))
    bbox.columns = ["image", "label", "x", "y", "w", "h"]

    examples = {"nih": [], "bbox": None, "pneumonia": []}
    wanted = [(f, lab) for f, lab in cfg["examples"]["nih"] if f in set(df["image"])]
    if len(wanted) < 4:  # sample config: take the first rows that exist
        wanted = [(r.image, r.finding_labels) for r in df.head(4).itertuples()]
    for fname, label in wanted:
        r = df[df["image"] == fname].iloc[0]
        meta = {"file": fname, "age": int(r["age_clean"]),
                "sex": r["gender"], "view": r["view"], "label": r["finding_labels"]}
        examples["nih"].append(nih_example(fname, r["image_folder"], label, meta))

    # One annotated case with its bounding box drawn.
    joined = bbox.merge(df[["image", "image_folder", "finding_labels"]], on="image")
    if len(joined):
        b = joined.iloc[0]
        examples["bbox"] = nih_example(
            b["image"], b["image_folder"], f"Bounding box: {b['label']}",
            {"file": b["image"], "label": b["finding_labels"],
             "box": f"x={b['x']:.0f}, y={b['y']:.0f}, w={b['w']:.0f}, h={b['h']:.0f}"},
            bbox=(b["x"], b["y"], b["w"], b["h"]),
        )

    # Pneumonia dataset: one normal, one pneumonia.
    for cls in ["NORMAL", "PNEUMONIA"]:
        d = pneu_root / "train" / cls
        f = next(iter(sorted(d.glob("*.jpeg"))), None) if d.is_dir() else None
        if f:
            examples["pneumonia"].append({
                "caption": cls.title(),
                "meta": {"file": f.name, "class": cls, "dataset": "pediatric"},
                "img": to_b64(Image.open(f)),
            })

    cols = ["image", "patient_id", "age_clean", "gender", "view", "followup",
            "width", "height", "finding_labels", "n_findings", "split",
            "image_folder", "has_bbox"]
    sample = df[cols].iloc[min(1, len(df) - 1)].to_dict()
    examples["sample_row"] = {k: (v.item() if hasattr(v, "item") else v)
                              for k, v in sample.items()}
    return examples


def run(cfg: Config) -> None:
    out = cfg.path("outputs") / "examples.json"
    with stage_run("examples", cfg) as rec:
        rec.inputs(cfg.processed("dataset.parquet"))
        examples = build(cfg)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(examples), encoding="utf-8")
        rec.outputs(out)
        rec.stats(nih_examples=len(examples["nih"]), kb=round(out.stat().st_size / 1024))
