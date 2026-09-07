#!/usr/bin/env python3
"""
FactLens startup script.
Run from the factlens/ directory:
    python run.py

Or with uvicorn directly:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload",
    ])
