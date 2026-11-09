"""Schémas pandera : colonnes, types et domaines attendus pour chaque table."""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column

IMAGE_NAME = r"^\d{8}_\d{3}\.png$"


def raw_metadata_schema() -> pa.DataFrameSchema:
    """Table clinique brute : on vérifie la forme, pas encore la plausibilité des âges."""
    return pa.DataFrameSchema(
        {
            "image": Column(str, Check.str_matches(IMAGE_NAME), unique=True, nullable=False),
            "finding_labels": Column(str, nullable=False),
            "followup": Column(int, Check.ge(0)),
            "patient_id": Column(int, Check.gt(0)),
            "age": Column(int, nullable=False),
            "gender": Column(str, Check.isin(["M", "F"])),
            "view": Column(str, Check.isin(["PA", "AP"])),
            "width": Column(int, Check.gt(0)),
            "height": Column(int, Check.gt(0)),
            "spacing_x": Column(float, Check.gt(0)),
            "spacing_y": Column(float, Check.gt(0)),
        },
        strict=True,
        coerce=False,
    )


def bbox_schema() -> pa.DataFrameSchema:
    return pa.DataFrameSchema({
        "image": Column(str, Check.str_matches(IMAGE_NAME)),
        "bbox_label": Column(str),
        "bbox_x": Column(float), "bbox_y": Column(float),
        "bbox_w": Column(float, Check.gt(0)), "bbox_h": Column(float, Check.gt(0)),
    }, strict=True)


def processed_schema(pathologies: list[str], age_min: int, age_max: int,
                     genders: list[str], views: list[str]) -> pa.DataFrameSchema:
    cols = {
        "image": Column(str, Check.str_matches(IMAGE_NAME), unique=True),
        "patient_id": Column(int, Check.gt(0)),
        "age_clean": Column(int, Check.in_range(age_min, age_max)),
        "gender": Column(str, Check.isin(genders)),
        "view": Column(str, Check.isin(views)),
        "followup": Column(int, Check.ge(0)),
        "split": Column(str, Check.isin(["train", "val", "test"])),
        "has_image": Column(int, Check.isin([0, 1])),
        "has_tabular": Column(int, Check.isin([0, 1])),
        "has_bbox": Column(int, Check.isin([0, 1])),
        "n_findings": Column(int, Check.in_range(0, len(pathologies))),
    }
    for p in pathologies:
        cols[f"lbl_{p}"] = Column(int, Check.isin([0, 1]))
    return pa.DataFrameSchema(cols, strict=False)
