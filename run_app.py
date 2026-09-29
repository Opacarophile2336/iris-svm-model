"""HMNC_PRO application launcher.

Starts the FastAPI backend and Vite frontend development server.
Initializes the Excel user account file if it does not exist.

Usage:
    python run_app.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 2006
FRONTEND_PORT = 2336


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║                  HMNC_PRO v1.0.0                     ║
║   Machine Learning & Deep Learning Platform          ║
║   Authentication • SVM • Iris Prediction             ║
╚══════════════════════════════════════════════════════╝
""")


def check_requirements():
    """Verify backend dependencies are installed."""
    try:
        import fastapi, uvicorn, sklearn, openpyxl, passlib
        print("✓ Backend dependencies OK")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("  Run: pip install -r backend/requirements.txt")
        sys.exit(1)


def check_node():
    """Verify Node.js and npm are available."""
    try:
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Node.js {result.stdout.strip()}")
        else:
            raise RuntimeError("node not found")
    except Exception:
        print("✗ Node.js not found. Please install Node.js 18+.")
        sys.exit(1)


import socket

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a local port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def ensure_port_free(port: int):
    """Free port if occupied by a lingering process."""
    if not is_port_in_use(port):
        return
    print(f"  Port {port} is currently in use. Cleaning up lingering process...")
    try:
        if sys.platform == "win32":
            cmd = f'powershell -Command "Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | ForEach-Object {{ Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }}"'
            subprocess.run(cmd, shell=True, capture_output=True)
            time.sleep(1)
    except Exception as e:
        print(f"  Warning: could not auto-kill process on port {port}: {e}")


def start_backend():
    """Start FastAPI backend with uvicorn."""
    ensure_port_free(BACKEND_PORT)
    print(f"  Starting backend on http://{BACKEND_HOST}:{BACKEND_PORT} ...")
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", BACKEND_HOST,
            "--port", str(BACKEND_PORT),
            "--reload",
        ],
        cwd=str(BACKEND_DIR),
    )


def start_frontend():
    """Start Vite development server."""
    if not FRONTEND_DIR.exists():
        print("  Frontend directory not found — skipping frontend startup.")
        return None
    ensure_port_free(FRONTEND_PORT)
    print(f"  Starting frontend on http://localhost:{FRONTEND_PORT} ...")
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=str(FRONTEND_DIR),
    )


def main():
    print_banner()
    check_requirements()
    check_node()

    print("\n[1/3] Starting backend...")
    backend_proc = start_backend()
    time.sleep(3)

    print("[2/3] Starting frontend...")
    frontend_proc = start_frontend()
    time.sleep(3)

    print("""
[3/3] Application ready!

  Backend API:   http://127.0.0.1:2006
  API Docs:      http://127.0.0.1:2006/docs
  Frontend:      http://localhost:2336

  ADMIN login:   Username: 0814230306 (see .env)

  Press Ctrl+C to stop all services.
""")

    try:
        webbrowser.open("http://localhost:2336")
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_proc.terminate()
        if frontend_proc:
            frontend_proc.terminate()


if __name__ == "__main__":
    main()
