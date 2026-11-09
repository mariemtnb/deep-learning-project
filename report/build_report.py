"""Génère report/rapport_tache3.pdf à partir de la source HTML (Edge ou Chrome headless).

    python report/build_report.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "msedge",
]


def browser() -> str:
    for c in CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    sys.exit("Edge ou Chrome introuvable.")


def main() -> None:
    meta = yaml.safe_load((HERE / "meta.yaml").read_text(encoding="utf-8"))
    html = (HERE / "rapport_tache3.html").read_text(encoding="utf-8")
    html = (html.replace("{{AUTEURS}}", meta["auteurs"]).replace("{{DEPOT}}", meta["depot"])
            .replace("{{DATE}}", meta["date"]))
    tmp = HERE / "_build.html"
    tmp.write_text(html, encoding="utf-8")
    out = HERE / "rapport_tache3.pdf"
    subprocess.run([browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", tmp.as_uri()], check=True, timeout=120)
    tmp.unlink()
    print(f"écrit : {out}")


if __name__ == "__main__":
    main()
