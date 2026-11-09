"""
Start the interface at http://localhost:8000

Run:  python app/run.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.server import app

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, threaded=True)
