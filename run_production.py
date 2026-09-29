"""IrisAI Studio — Production One-Port Application Launcher.

Starts the FastAPI production server serving both the REST API and the
compiled React production build on a single local port (default: 2336).

Optionally exposes the single port via Cloudflare Quick Tunnel when requested with --tunnel.

Usage:
    python run_production.py [--port 2336] [--host 127.0.0.1] [--tunnel] [--build]
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"

# Ensure backend directory is in sys.path
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def print_banner(host: str, port: int, tunnel_requested: bool):
    print("=" * 64)
    print("  IrisAI Studio — One-Port Production Server")
    print("  Intelligent Iris Classification & Machine Learning Platform")
    print("=" * 64)
    print(f"  Local Host:           {host}")
    print(f"  Production Port:      {port}")
    print(f"  Single-Port Mode:     Active (API + SPA UI + Swagger + Static)")
    print(f"  Public Tunnel:        {'Enabled (Quick Tunnel)' if tunnel_requested else 'Disabled (Local only)'}")
    print("=" * 64)


def check_python_dependencies() -> bool:
    """Verify essential backend dependencies are installed."""
    required = ["fastapi", "uvicorn", "sklearn", "openpyxl", "sqlalchemy", "pyodbc"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  [ERROR] Missing required Python packages: {', '.join(missing)}")
        print("  Please install with: pip install -r backend/requirements.txt")
        return False
    print("  [OK] Python dependencies verified.")
    return True


def check_frontend_build(auto_build: bool = False) -> bool:
    """Verify frontend production build exists."""
    index_html = FRONTEND_DIST_DIR / "index.html"
    assets_dir = FRONTEND_DIST_DIR / "assets"

    if index_html.is_file() and assets_dir.is_dir():
        print("  [OK] Frontend production build verified.")
        return True

    print("  [WARNING] Frontend production build (frontend/dist) is missing or incomplete.")

    if auto_build:
        print("  Building frontend now (npm run build)...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        try:
            res = subprocess.run([npm_cmd, "run", "build"], cwd=str(FRONTEND_DIR), check=True)
            if index_html.is_file():
                print("  [OK] Frontend built successfully.")
                return True
        except Exception as e:
            print(f"  [ERROR] Frontend build failed: {e}")
            return False

    print("  To build the frontend, run:")
    print("    cd frontend && npm run build")
    print("  Or run this launcher with the --build flag:")
    print("    python run_production.py --build")
    return False


def check_database_connectivity() -> bool:
    """Validate SQL Server connectivity using read-only probe without credential exposure."""
    try:
        from database.connection import engine
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("  [OK] SQL Server connection verified (VICTUS2336\\SQLEXPRESS01).")
        return True
    except Exception:
        print("  [ERROR] Could not connect to SQL Server. Please verify the database service is running.")
        return False


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def start_tunnel(port: int, stop_event: threading.Event) -> tuple[subprocess.Popen | None, str | None]:
    """Start Cloudflare Quick Tunnel forwarding strictly to http://127.0.0.1:port."""
    try:
        ver_check = subprocess.run(["cloudflared", "--version"], capture_output=True, text=True, timeout=5)
        if ver_check.returncode != 0:
            print("  [WARNING] cloudflared is not available. Skipping tunnel.")
            return None, None
    except Exception:
        print("  [WARNING] cloudflared command not found on PATH. Skipping tunnel.")
        return None, None

    tunnel_url_target = f"http://127.0.0.1:{port}"
    print(f"  Starting Cloudflare Quick Tunnel forwarding strictly to {tunnel_url_target}...")

    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", tunnel_url_target],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        print(f"  [ERROR] Failed to spawn cloudflared process: {e}")
        return None, None

    tunnel_url = None
    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")

    # Read output to capture the trycloudflare.com URL
    start_time = time.time()
    while time.time() - start_time < 20 and proc.poll() is None and not stop_event.is_set():
        line = proc.stderr.readline()
        if not line:
            time.sleep(0.1)
            continue
        match = url_pattern.search(line)
        if match:
            tunnel_url = match.group(0)
            break

    if tunnel_url:
        print(f"  [OK] Public Quick Tunnel active: {tunnel_url}")
        print("  (Note: This URL is temporary and provided by Cloudflare trycloudflare.com)")
    else:
        print("  [WARNING] Could not obtain Cloudflare Tunnel URL within 20s. Local server remains active.")

    return proc, tunnel_url


def main():
    parser = argparse.ArgumentParser(description="IrisAI Studio One-Port Production Launcher")
    parser.add_argument("--port", type=int, default=2336, help="Port to bind production server (default: 2336)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--tunnel", action="store_true", help="Start Cloudflare Quick Tunnel for public access")
    parser.add_argument("--build", action="store_true", help="Build frontend automatically if missing")
    args = parser.parse_args()

    port = args.port
    host = args.host
    tunnel_requested = args.tunnel

    print_banner(host, port, tunnel_requested)

    # 1. Environment and dependency checks
    if not check_python_dependencies():
        sys.exit(1)

    # 2. Database check
    if not check_database_connectivity():
        sys.exit(1)

    # 3. Frontend production build check
    if not check_frontend_build(auto_build=args.build):
        print("  [ERROR] Aborting production launch because frontend build is missing.")
        sys.exit(1)

    # 4. Port conflict check (Never auto-kill unknown processes!)
    if is_port_in_use(port, host):
        print(f"  [ERROR] Port {port} is already in use by another process on {host}.")
        print("  Please terminate the conflicting process or specify a different port with --port.")
        sys.exit(1)

    # 5. Prepare uvicorn server
    print(f"\n  Starting Uvicorn production server on http://{host}:{port} ...")
    uvicorn_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", host,
            "--port", str(port),
        ],
        cwd=str(BACKEND_DIR),
    )

    tunnel_proc = None
    tunnel_url = None
    stop_event = threading.Event()

    # 6. Start Cloudflare Tunnel if requested
    if tunnel_requested:
        tunnel_proc, tunnel_url = start_tunnel(port, stop_event)

    print("\n" + "=" * 64)
    print("  APPLICATION READY IN PRODUCTION MODE!")
    print(f"  Local Web Application: http://{host}:{port}/")
    print(f"  Interactive API Docs:  http://{host}:{port}/docs")
    print(f"  OpenAPI Specification: http://{host}:{port}/openapi.json")
    if tunnel_url:
        print(f"  Public Tunnel URL:     {tunnel_url}")
    print("=" * 64)
    print("  Press Ctrl+C to gracefully shut down the production server.\n")

    try:
        uvicorn_proc.wait()
    except KeyboardInterrupt:
        print("\n  Received shutdown signal. Stopping services...")
    finally:
        stop_event.set()
        if tunnel_proc and tunnel_proc.poll() is None:
            print("  Stopping Cloudflare Tunnel...")
            tunnel_proc.terminate()
            try:
                tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                tunnel_proc.kill()

        if uvicorn_proc and uvicorn_proc.poll() is None:
            print("  Stopping Uvicorn server...")
            uvicorn_proc.terminate()
            try:
                uvicorn_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                uvicorn_proc.kill()

        print("  Shutdown complete. Port released.")


if __name__ == "__main__":
    main()
