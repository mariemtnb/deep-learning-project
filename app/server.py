"""
Backend for the project interface: add datasets from a link or an upload, inspect
them, treat tabular files, and download the results.

Run:  python app/run.py
"""
from __future__ import annotations

import io
import ipaddress
import json
import re
import shutil
import socket
import threading
import time
import uuid
import zipfile
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import pandas as pd
import requests
from flask import Flask, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
# Table produite par le pipeline (dvc repro / dvc pull). Les copies traitées vont à côté.
PROCESSED = ROOT / "data" / "processed"
STORE = ROOT / "data_store"
DIST = ROOT / "app" / "web" / "dist"
STORE.mkdir(exist_ok=True)

MAX_DOWNLOAD = 5 * 1024**3
MAX_UNZIPPED = 20 * 1024**3
TABULAR = {".csv", ".tsv", ".parquet", ".json"}
BUILTIN = "nih-processed"

app = Flask(__name__, static_folder=None)
meta_lock = threading.Lock()


class HTTPException(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail


# --------------------------------------------------------------------------- #
# Dataset storage
# --------------------------------------------------------------------------- #
def dataset_dir(ds_id: str) -> Path:
    if ds_id == BUILTIN:
        return PROCESSED
    if not re.fullmatch(r"[0-9a-f]{12}", ds_id) or not (STORE / ds_id).is_dir():
        raise HTTPException(404, "Dataset not found")
    return STORE / ds_id


def read_meta(ds_id: str) -> dict:
    if ds_id == BUILTIN:
        return {"id": BUILTIN, "name": "NIH processed metadata", "source": "Project pipeline",
                "status": "ready", "builtin": True, "bytes": 0, "error": None}
    return json.loads((dataset_dir(ds_id) / "meta.json").read_text(encoding="utf-8"))


def write_meta(ds_id: str, **changes) -> None:
    path = STORE / ds_id / "meta.json"
    with meta_lock:
        meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        meta.update(changes)
        path.write_text(json.dumps(meta), encoding="utf-8")


def create_dataset(name: str, source: str, status: str) -> str:
    ds_id = uuid.uuid4().hex[:12]
    (STORE / ds_id / "files").mkdir(parents=True)
    write_meta(ds_id, id=ds_id, name=name, source=source, status=status, bytes=0,
               error=None, created=time.time(), builtin=False)
    return ds_id


def files_root(ds_id: str) -> Path:
    return PROCESSED if ds_id == BUILTIN else dataset_dir(ds_id) / "files"


def list_files(ds_id: str) -> list[dict]:
    root = files_root(ds_id)
    if ds_id == BUILTIN:
        paths = [p for p in root.glob("dataset.parquet") if p.is_file()]
        paths += [p for p in root.glob("*__treated_*") if p.is_file()]
    else:
        paths = [p for p in root.rglob("*") if p.is_file()]
    return [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
             "tabular": p.suffix.lower() in TABULAR} for p in sorted(paths)]


def safe_file(ds_id: str, rel: str) -> Path:
    root = files_root(ds_id).resolve()
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "File not found")
    if rel not in {f["path"] for f in list_files(ds_id)}:
        raise HTTPException(404, "File not found")
    return path


def clean_filename(name: str) -> str:
    name = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip(" .")
    return name or "download.bin"


def extract_zip(path: Path) -> None:
    """Unpack next to the archive, refusing path escapes and oversized archives."""
    target = path.parent.resolve()
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        if sum(i.file_size for i in infos) > MAX_UNZIPPED:
            raise ValueError("Archive is larger than 20 GB once unpacked")
        for info in infos:
            dest = (target / info.filename).resolve()
            if not dest.is_relative_to(target):
                raise ValueError(f"Unsafe path in archive: {info.filename}")
        zf.extractall(target)
    path.unlink()


# --------------------------------------------------------------------------- #
# Download from a link
# --------------------------------------------------------------------------- #
def check_public_url(url: str) -> None:
    """Only public http(s) hosts, so a link cannot reach this machine or its network."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("The link must start with http:// or https://")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        raise ValueError("That website could not be found")
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise ValueError("Links to private or local addresses are not allowed")


def open_link(url: str) -> requests.Response:
    for _ in range(6):
        check_public_url(url)
        resp = requests.get(url, stream=True, timeout=30, allow_redirects=False,
                            headers={"User-Agent": "xray-project/1.0"})
        if resp.is_redirect:
            url = urljoin(url, resp.headers["location"])
            resp.close()
            continue
        return resp
    raise ValueError("Too many redirects")


def download_job(ds_id: str, url: str) -> None:
    try:
        resp = open_link(url)
        if resp.status_code != 200:
            raise ValueError(f"The website answered with error {resp.status_code}")
        if "text/html" in resp.headers.get("content-type", ""):
            raise ValueError("This link opens a web page, not a file. Use the direct "
                             "download link. Sites such as Kaggle need you to sign in, so "
                             "download the file yourself and use Upload.")
        match = re.search(r'filename="?([^";]+)', resp.headers.get("content-disposition", ""))
        name = clean_filename(match.group(1) if match else unquote(urlparse(resp.url).path))
        dest = files_root(ds_id) / name
        size, last = 0, 0.0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1024 * 1024):
                size += len(chunk)
                if size > MAX_DOWNLOAD:
                    raise ValueError("File is larger than the 5 GB limit")
                fh.write(chunk)
                if time.time() - last > 1:
                    write_meta(ds_id, bytes=size)
                    last = time.time()
        write_meta(ds_id, bytes=size)
        finish_import(ds_id, dest)
    except Exception as e:
        shutil.rmtree(files_root(ds_id), ignore_errors=True)
        files_root(ds_id).mkdir(exist_ok=True)
        write_meta(ds_id, status="failed", error=str(e))


def finish_import(ds_id: str, path: Path) -> None:
    if zipfile.is_zipfile(path):
        write_meta(ds_id, status="unpacking")
        extract_zip(path)
    write_meta(ds_id, status="ready")


# --------------------------------------------------------------------------- #
# Tabular files
# --------------------------------------------------------------------------- #
def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    try:
        if ext == ".parquet":
            return pd.read_parquet(path)
        if ext == ".json":
            return pd.read_json(path)
        return pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", low_memory=False)
    except Exception as e:
        raise HTTPException(422, f"This file could not be read as a table: {e}")


def profile(df: pd.DataFrame) -> dict:
    columns = []
    for c in df.columns:
        s = df[c]
        item = {"name": str(c), "type": str(s.dtype), "missing": int(s.isna().sum()),
                "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            item.update(min=_num(s.min()), max=_num(s.max()), mean=_num(s.mean()))
        columns.append(item)
    preview = df.head(8).astype(object).where(df.head(8).notna(), None)
    return {"rows": len(df), "cols": df.shape[1], "duplicates": int(df.duplicated().sum()),
            "missing_cells": int(df.isna().sum().sum()), "columns": columns,
            "preview": json.loads(preview.to_json(orient="records", default_handler=str))}


def _num(v) -> float | None:
    return None if pd.isna(v) else round(float(v), 3)


def apply_op(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, str]:
    kind = op.get("op")
    cols = [c for c in op.get("columns", []) if c in df.columns] or list(df.columns)
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])
               and not pd.api.types.is_bool_dtype(df[c])]
    n = len(df)
    if kind == "drop_duplicates":
        df = df.drop_duplicates()
        return df, f"Removed {n - len(df):,} duplicate rows"
    if kind == "drop_empty_rows":
        df = df.dropna(subset=cols)
        return df, f"Removed {n - len(df):,} rows with missing values"
    if kind == "drop_columns":
        chosen = [c for c in op.get("columns", []) if c in df.columns]
        return df.drop(columns=chosen), f"Removed {len(chosen)} columns"
    if kind == "fill_missing":
        before = int(df[cols].isna().sum().sum())
        for c in cols:
            if c in numeric:
                df[c] = df[c].fillna(df[c].median())
            elif not df[c].mode().empty:
                df[c] = df[c].fillna(df[c].mode().iloc[0])
        return df, f"Filled {before - int(df[cols].isna().sum().sum()):,} missing values"
    if kind == "clean_text":
        text = [c for c in cols if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])]
        for c in text:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())
        return df, f"Trimmed spaces in {len(text)} text columns"
    if kind == "clip_outliers":
        clipped = 0
        for c in numeric:
            q1, q3 = df[c].quantile([0.25, 0.75])
            low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            clipped += int(((df[c] < low) | (df[c] > high)).sum())
            df[c] = df[c].astype(float).clip(low, high)
        return df, f"Capped {clipped:,} extreme values"
    if kind == "normalise":
        for c in numeric:
            low, high = df[c].min(), df[c].max()
            df[c] = 0.0 if low == high else (df[c] - low) / (high - low)
        return df, f"Rescaled {len(numeric)} numeric columns to 0..1"
    raise HTTPException(400, f"Unknown treatment: {kind}")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.errorhandler(HTTPException)
def http_error(e: HTTPException):
    return jsonify(detail=e.detail), e.status


@app.get("/api/summary")
def summary():
    return json.loads((OUT / "summary.json").read_text(encoding="utf-8"))


@app.get("/api/datasets")
def datasets():
    ids = [BUILTIN] + [p.name for p in STORE.iterdir() if (p / "meta.json").exists()]
    rows = [{**read_meta(i), "n_files": len(list_files(i))} for i in ids]
    return jsonify(sorted(rows, key=lambda m: -m.get("created", 9e12)))


@app.post("/api/datasets/link")
def add_link():
    body = request.get_json()
    url, name = body.get("url", ""), body.get("name", "").strip()
    try:
        check_public_url(url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    name = name or Path(unquote(urlparse(url).path)).name or url
    ds_id = create_dataset(name, url, "downloading")
    threading.Thread(target=download_job, args=(ds_id, url), daemon=True).start()
    return {"id": ds_id}, 202


@app.post("/api/datasets/upload")
def upload():
    if "file" not in request.files:
        raise HTTPException(400, "No file was sent")
    file = request.files["file"]
    filename = clean_filename(file.filename or "upload.bin")
    ds_id = create_dataset(request.form.get("name", "").strip() or filename,
                           "Uploaded file", "unpacking")
    dest = files_root(ds_id) / filename
    file.save(dest)
    write_meta(ds_id, bytes=dest.stat().st_size)
    try:
        finish_import(ds_id, dest)
    except Exception as e:
        write_meta(ds_id, status="failed", error=str(e))
    return {"id": ds_id}, 201


@app.get("/api/datasets/<ds_id>")
def dataset(ds_id: str):
    files = list_files(ds_id)
    return {**read_meta(ds_id), "n_files": len(files), "files": files[:500],
            "files_cut": len(files) > 500}


@app.get("/api/datasets/<ds_id>/profile")
def dataset_profile(ds_id: str):
    return profile(load_table(safe_file(ds_id, request.args.get("file", ""))))


@app.post("/api/datasets/<ds_id>/treat")
def treat(ds_id: str):
    body = request.get_json()
    source = safe_file(ds_id, body.get("file", ""))
    df = load_table(source)
    log = []
    for op in body.get("ops", []):
        df, line = apply_op(df, op)
        log.append(line)
    stem = re.sub(r"__treated_\d+$", "", source.stem)
    n = 1
    while (source.parent / f"{stem}__treated_{n}.csv").exists():
        n += 1
    out = source.parent / f"{stem}__treated_{n}.csv"
    df.to_csv(out, index=False)
    return {"file": out.relative_to(files_root(ds_id).resolve()).as_posix(), "log": log,
            "profile": profile(df)}


@app.get("/api/datasets/<ds_id>/download")
def download(ds_id: str):
    file = request.args.get("file", "")
    if file:
        return send_file(safe_file(ds_id, file), as_attachment=True)
    if ds_id == BUILTIN:
        return send_file(PROCESSED / "dataset.parquet", as_attachment=True)
    temp = STORE / f"{ds_id}-{uuid.uuid4().hex}.zip"
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in list_files(ds_id):
            zf.write(files_root(ds_id) / f["path"], f["path"])
    data = io.BytesIO(temp.read_bytes()) if temp.stat().st_size < 2**31 else None
    if data is None:
        return send_file(temp, as_attachment=True,
                         download_name=f"{clean_filename(read_meta(ds_id)['name'])}.zip")
    temp.unlink()
    return send_file(data, as_attachment=True,
                     download_name=f"{clean_filename(read_meta(ds_id)['name'])}.zip")


@app.delete("/api/datasets/<ds_id>")
def delete(ds_id: str):
    if ds_id == BUILTIN:
        raise HTTPException(400, "The project dataset cannot be deleted")
    shutil.rmtree(dataset_dir(ds_id))
    return {"ok": True}


@app.get("/report")
def report():
    return send_file(OUT / "dashboard.html")


@app.get("/")
@app.get("/<path:path>")
def web(path: str = "index.html"):
    return send_from_directory(DIST, path)
