"""
Data pipeline for the medical-diagnosis project (NIH ChestX-ray14 + Pneumonia).

Steps:
  1. Load the NIH clinical metadata (Data_Entry_2017.csv).
  2. Clean it (fix columns, repair implausible ages, normalise categoricals).
  3. Feature engineering (14 pathology one-hot labels, derived features).
  4. Attach the official train/val/test split by PATIENT (no data leakage).
  5. Index the image files across images_001..012 and attach bbox availability.
  6. Catalogue the separate "Chest X-Ray Pneumonia" image dataset.
  7. Export processed tables + a summary JSON used by the dashboard.

Run:  python src/pipeline.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "project deep learning datasets"
NIH = DATA / "NIH Chest X-rays"
PNEU = DATA / "Chest X-Ray Images (Pneumonia)" / "chest_xray" / "chest_xray"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# The 14 pathology classes in NIH ChestX-ray14.
PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def log(msg: str) -> None:
    print(f"[pipeline] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# 1-2. Load & clean clinical metadata
# --------------------------------------------------------------------------- #
def load_clean_metadata() -> pd.DataFrame:
    df = pd.read_csv(NIH / "Data_Entry_2017.csv")

    # The CSV has a trailing empty column from a dangling comma in the header.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    df = df.rename(columns={
        "Image Index": "image",
        "Finding Labels": "finding_labels",
        "Follow-up #": "followup",
        "Patient ID": "patient_id",
        "Patient Age": "age",
        "Patient Gender": "gender",
        "View Position": "view",
        "OriginalImage[Width,Height]": "width",
        "Height]": "height",  # defensive; real name resolved below
        "OriginalImagePixelSpacing[x,y]": "spacing_x",
        "y]": "spacing_y",
    })
    # Column names for the bracketed pairs vary; resolve positionally instead.
    cols = list(df.columns)
    # After rename the two numeric pairs sit at known positions; map by order.
    rename_by_pos = {}
    for i, c in enumerate(cols):
        if c in ("OriginalImage[Width", "width"):
            rename_by_pos[c] = "width"
        elif c in ("Height]",):
            rename_by_pos[c] = "height"
        elif c in ("OriginalImagePixelSpacing[x", "spacing_x"):
            rename_by_pos[c] = "spacing_x"
        elif c in ("y]",):
            rename_by_pos[c] = "spacing_y"
    df = df.rename(columns=rename_by_pos)

    # --- Age cleaning -------------------------------------------------------
    # NIH ages are integers but contain data-entry errors (e.g. 148, 411).
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_invalid"] = (df["age"] <= 0) | (df["age"] > 100)
    # Replace implausible ages with the median of the valid ones.
    valid_median = int(df.loc[~df["age_invalid"], "age"].median())
    df["age_clean"] = df["age"].where(~df["age_invalid"], valid_median).astype(int)

    # --- Categorical normalisation -----------------------------------------
    df["gender"] = df["gender"].str.upper().str.strip()
    df["view"] = df["view"].str.upper().str.strip()

    # --- Numeric coercion ---------------------------------------------------
    for c in ["width", "height", "spacing_x", "spacing_y", "followup"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    log(f"loaded {len(df):,} rows, {df['patient_id'].nunique():,} patients")
    return df


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["finding_labels"].fillna("").str.split("|")

    # One-hot encode the 14 pathologies.
    for p in PATHOLOGIES:
        df[f"lbl_{p}"] = labels.apply(lambda ls, p=p: int(p in ls))

    df["is_normal"] = (df["finding_labels"] == "No Finding").astype(int)
    df["n_findings"] = df[[f"lbl_{p}" for p in PATHOLOGIES]].sum(axis=1)
    df["is_multilabel"] = (df["n_findings"] > 1).astype(int)

    # Age bands for stratification / dashboard.
    df["age_group"] = pd.cut(
        df["age_clean"],
        bins=[0, 18, 30, 45, 60, 75, 120],
        labels=["0-18", "19-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    # Image geometry feature.
    df["aspect_ratio"] = (df["width"] / df["height"]).round(3)
    df["is_followup"] = (df["followup"] > 0).astype(int)

    return df


# --------------------------------------------------------------------------- #
# 4. Patient-level split (no leakage)
# --------------------------------------------------------------------------- #
def attach_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    test = set((NIH / "test_list.txt").read_text().split())
    trainval = set((NIH / "train_val_list.txt").read_text().split())

    df["split"] = np.where(
        df["image"].isin(test), "test",
        np.where(df["image"].isin(trainval), "trainval", "unlisted"),
    )

    # Carve a validation set out of trainval BY PATIENT so no patient appears
    # in both train and val (prevents data leakage).
    tv_patients = df.loc[df["split"] == "trainval", "patient_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(tv_patients)
    n_val = int(len(tv_patients) * 0.1)
    val_patients = set(tv_patients[:n_val])

    df.loc[(df["split"] == "trainval") & df["patient_id"].isin(val_patients), "split"] = "val"
    df.loc[df["split"] == "trainval", "split"] = "train"

    # Sanity: no patient shared across splits.
    overlap = (
        df.groupby("patient_id")["split"].nunique().gt(1).sum()
    )
    log(f"split sizes: {df['split'].value_counts().to_dict()}")
    log(f"patients appearing in >1 split: {overlap} (0 = no leakage)")
    return df


# --------------------------------------------------------------------------- #
# 5. Image index + bbox
# --------------------------------------------------------------------------- #
def index_images(df: pd.DataFrame) -> pd.DataFrame:
    folder_of = {}
    for sub in sorted(NIH.glob("images_*")):
        img_dir = sub / "images"
        if img_dir.is_dir():
            for f in img_dir.iterdir():
                folder_of[f.name] = sub.name
    df["image_folder"] = df["image"].map(folder_of)
    df["image_found"] = df["image_folder"].notna().astype(int)
    log(f"images located on disk: {df['image_found'].sum():,} / {len(df):,}")

    bbox = pd.read_csv(NIH / "BBox_List_2017.csv")
    bbox = bbox.loc[:, ~bbox.columns.str.startswith("Unnamed")]
    bbox_images = set(bbox["Image Index"])
    df["has_bbox"] = df["image"].isin(bbox_images).astype(int)
    log(f"images with bounding boxes: {df['has_bbox'].sum():,}")
    return df


# --------------------------------------------------------------------------- #
# 6. Pneumonia image dataset catalogue
# --------------------------------------------------------------------------- #
def catalogue_pneumonia() -> dict:
    counts = {}
    if not PNEU.exists():
        return counts
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            d = PNEU / split / cls
            n = sum(1 for _ in d.glob("*.jpeg")) if d.exists() else 0
            counts[f"{split}/{cls}"] = n
    log(f"pneumonia dataset catalogued: {sum(counts.values()):,} images")
    return counts


# --------------------------------------------------------------------------- #
# 7. Summary + export
# --------------------------------------------------------------------------- #
def build_summary(df: pd.DataFrame, pneu: dict) -> dict:
    pcols = [f"lbl_{p}" for p in PATHOLOGIES]
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_images": int(len(df)),
        "n_patients": int(df["patient_id"].nunique()),
        "n_images_on_disk": int(df["image_found"].sum()),
        "age": {
            "n_invalid_repaired": int(df["age_invalid"].sum()),
            "mean_clean": round(float(df["age_clean"].mean()), 1),
            "median_clean": int(df["age_clean"].median()),
        },
        "gender_counts": df["gender"].value_counts().to_dict(),
        "view_counts": df["view"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "normal_vs_abnormal": {
            "normal": int(df["is_normal"].sum()),
            "abnormal": int((~df["is_normal"].astype(bool)).sum()),
        },
        "multilabel_counts": df["n_findings"].value_counts().sort_index().to_dict(),
        "pathology_counts": {p: int(df[f"lbl_{p}"].sum()) for p in PATHOLOGIES},
        "age_group_counts": df["age_group"].value_counts().to_dict(),
        "pathology_by_gender": {
            p: df.groupby("gender")[f"lbl_{p}"].sum().to_dict() for p in PATHOLOGIES
        },
        "has_bbox": int(df["has_bbox"].sum()),
        "pneumonia_dataset": pneu,
    }
    # JSON-safe (numpy -> python).
    return json.loads(json.dumps(summary, default=int))


def main() -> None:
    df = load_clean_metadata()
    df = engineer_features(df)
    df = attach_split(df)
    df = index_images(df)
    pneu = catalogue_pneumonia()
    summary = build_summary(df, pneu)

    # Processed table (keep the useful columns).
    keep = (
        ["image", "patient_id", "age_clean", "age_invalid", "age_group",
         "gender", "view", "followup", "is_followup", "width", "height",
         "aspect_ratio", "finding_labels", "is_normal", "n_findings",
         "is_multilabel", "split", "image_folder", "image_found", "has_bbox"]
        + [f"lbl_{p}" for p in PATHOLOGIES]
    )
    out_csv = OUT / "processed_metadata.csv"
    df[keep].to_csv(out_csv, index=False)
    try:
        df[keep].to_parquet(OUT / "processed_metadata.parquet", index=False)
    except Exception as e:  # parquet engine optional
        log(f"parquet skipped ({e})")

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_csv}")
    log(f"wrote {OUT / 'summary.json'}")
    log("done.")


if __name__ == "__main__":
    main()
