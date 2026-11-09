"""Observabilité : logs structurés (JSON lines) et fiche de traçabilité par étape.

Chaque étape écrit `reports/stages/<etape>.json` avec : volumes en entrée/sortie,
durée, empreintes MD5 des fichiers lus et écrits, commit Git et configuration utilisée.
C'est ce qui permet de remonter à l'origine d'un fichier produit.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .config import ROOT, Config

# Python 3.13 sur Windows : asyncio.current_task() appelé par logging peut planter
# quand asyncio a été importé par une dépendance sans boucle active. On n'en a pas besoin.
logging.logAsyncioTasks = False

_LOGGERS: dict[str, logging.Logger] = {}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "stage": record.name.removeprefix("pipeline."),
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, ensure_ascii=False)


def get_logger(stage: str, cfg: Config) -> logging.Logger:
    name = f"pipeline.{stage}"
    if name in _LOGGERS:
        return _LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(f"[{stage}] %(message)s"))
    logger.addHandler(console)

    log_dir = cfg.path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8")
    jsonl.setFormatter(_JsonFormatter())
    logger.addHandler(jsonl)

    _LOGGERS[name] = logger
    return logger


def log_kv(logger: logging.Logger, msg: str, **fields) -> None:
    """Message lisible en console + champs structurés dans le fichier JSONL."""
    logger.info(msg, extra={"extra_fields": fields})


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@contextmanager
def stage_run(stage: str, cfg: Config):
    """Contexte d'exécution d'une étape : chronométrage + fiche de traçabilité.

    Utilisation :
        with stage_run("preprocess", cfg) as run:
            run.inputs(path_a); ...; run.outputs(path_b); run.stats(rows=...)
    """
    logger = get_logger(stage, cfg)
    record = _StageRecord(stage, cfg, logger)
    start = time.perf_counter()
    log_kv(logger, "début", event="start")
    try:
        yield record
        record.data["status"] = "ok"
    except Exception as exc:
        record.data["status"] = "failed"
        record.data["error"] = f"{type(exc).__name__}: {exc}"
        log_kv(logger, f"échec : {exc}", event="failed")
        raise
    finally:
        record.data["duration_s"] = round(time.perf_counter() - start, 2)
        record.write()
        log_kv(logger, f"fin ({record.data['status']}, {record.data['duration_s']} s)",
               event="end", status=record.data["status"])


class _StageRecord:
    def __init__(self, stage: str, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        self.data: dict = {
            "stage": stage,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": git_commit(),
            "config": _rel(cfg.source),
            "inputs": {},
            "outputs": {},
            "stats": {},
        }

    def inputs(self, *paths: Path) -> None:
        for p in paths:
            self.data["inputs"][_rel(p)] = md5(p) if p.is_file() else "directory"

    def outputs(self, *paths: Path) -> None:
        for p in paths:
            self.data["outputs"][_rel(p)] = md5(p) if p.is_file() else "directory"

    def stats(self, **values) -> None:
        self.data["stats"].update(values)
        log_kv(self.logger, ", ".join(f"{k}={v}" for k, v in values.items()), **values)

    def info(self, msg: str, **fields) -> None:
        log_kv(self.logger, msg, **fields)

    def write(self) -> None:
        out = self.cfg.report(f"stages/{self.data['stage']}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.data, indent=2, ensure_ascii=False, default=str),
                       encoding="utf-8")
