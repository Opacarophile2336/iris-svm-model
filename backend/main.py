"""HMNC_PRO — FastAPI main application."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Ensure backend directory is in sys.path regardless of launch directory
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse, JSONResponse

import config
from storage.excel_repository import init_excel_file
from ml.svm_service import train_all_kernels, load_best_model
from image.species_image import warm_cache

from auth.router import router as auth_router
from ml.router import router as ml_router
from prediction.router import router as prediction_router
from image.router import router as image_router
from datasets.router import router as datasets_router
from dl.router import router as dl_router
from translation.router import router as translation_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    logger.info("=== HMNC_PRO Starting ===")

    # 1. Initialize Excel file (creates if missing, preserves if exists)
    init_excel_file()
    logger.info(f"Excel accounts file: {config.USER_ACCOUNTS_PATH}")

    # 2. Load or train best SVM model
    if not load_best_model():
        logger.info("No trained model found. Training SVM (RBF, Linear, Poly)...")
        try:
            results = train_all_kernels(save=True)
            best = results.get("best_kernel", "unknown")
            logger.info(f"Training complete. Best kernel: {best.upper()}")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    else:
        logger.info("Loaded existing trained model.")

    # 3. Pre-warm image cache (non-blocking background)
    try:
        warm_cache()
        logger.info("Species image cache warmed.")
    except Exception as e:
        logger.warning(f"Image cache warm failed (non-fatal): {e}")

    logger.info("=== HMNC_PRO Ready ===")
    yield
    logger.info("=== HMNC_PRO Shutting Down ===")


app = FastAPI(
    title="HMNC_PRO API",
    description="Machine Learning & Deep Learning Platform with Role-Based Access",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server and Cloudflare Tunnel origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:2336",
        "http://127.0.0.1:2336",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:2006",
        "http://127.0.0.1:2006",
    ],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|.*\.trycloudflare\.com)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Register all existing API routers first
app.include_router(auth_router)
app.include_router(ml_router)
app.include_router(prediction_router)
app.include_router(image_router)
app.include_router(datasets_router)
app.include_router(dl_router)
app.include_router(translation_router, prefix="/translation", tags=["Translation"])


@app.get("/health")
def health():
    return {"status": "ok"}


# 2. Static Asset Resolution under /assets/{file_path:path}
# Prioritizes frontend build assets (JS, CSS), falls back to Iris specimen images
@app.get("/assets/{file_path:path}")
async def serve_asset(file_path: str):
    # Path traversal protection
    if ".." in file_path or file_path.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid asset path.")

    # A. Check frontend/dist/assets/ (production JS/CSS bundles)
    dist_assets_dir = config.FRONTEND_DIST_DIR / "assets"
    if dist_assets_dir.is_dir():
        candidate = (dist_assets_dir / file_path).resolve()
        try:
            if candidate.is_file() and candidate.is_relative_to(dist_assets_dir.resolve()):
                return FileResponse(str(candidate))
        except (ValueError, RuntimeError):
            pass

    # B. Fallback to assets/images/ (authentic botanical specimen images)
    images_dir = config.ASSETS_DIR
    if images_dir.is_dir():
        candidate = (images_dir / file_path).resolve()
        try:
            if candidate.is_file() and candidate.is_relative_to(images_dir.resolve()):
                return FileResponse(str(candidate))
        except (ValueError, RuntimeError):
            pass

    # Asset strictly not found — return 404 (NEVER return HTML for assets)
    raise HTTPException(status_code=404, detail=f"Asset '{file_path}' not found.")


# Reserved prefixes that SPA fallback MUST NEVER intercept
RESERVED_PREFIXES = (
    "/auth",
    "/ml",
    "/prediction",
    "/image",
    "/datasets",
    "/dl",
    "/translation",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/assets",
    "/health",
)

# Common static file extensions that must return 404 if missing, never index.html
STATIC_EXTENSIONS = (
    ".js", ".css", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".json", ".map", ".woff", ".woff2", ".ttf", ".eot", ".txt", ".webp"
)


# 3. Root Endpoint
@app.get("/")
async def root():
    index_file = config.FRONTEND_DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))
    return {
        "name": "HMNC_PRO",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "message": "Development mode — frontend production build not found. Use Vite on port 2336 or run 'npm run build'.",
    }


# 4. SPA Catch-All Route for Frontend Navigation
# Note: Certain frontend page routes (/prediction, /datasets) share base prefix names with API
# routers but are distinct client-side pages when requested without an API sub-path.
EXACT_FRONTEND_PAGE_ROUTES = {"/prediction", "/datasets"}


@app.get("/{full_path:path}")
async def spa_fallback(full_path: str):
    norm_path = "/" + full_path.lstrip("/")

    # Check path traversal
    if ".." in full_path or "%2e" in full_path.lower():
        raise HTTPException(status_code=400, detail="Invalid path.")

    # 1. Check if a static file in frontend/dist exists (e.g. favicon.svg, icons.svg)
    if config.FRONTEND_DIST_DIR.is_dir():
        candidate = (config.FRONTEND_DIST_DIR / full_path).resolve()
        try:
            if candidate.is_file() and candidate.is_relative_to(config.FRONTEND_DIST_DIR.resolve()):
                return FileResponse(str(candidate))
        except (ValueError, RuntimeError):
            pass

    # 2. If the request has a file extension (e.g. .py, .js, .css, .png) and was not found in dist,
    # it is a missing static/asset file. Strictly return 404, NEVER return index.html!
    file_name = Path(full_path).name
    if "." in file_name:
        raise HTTPException(status_code=404, detail="File not found.")

    # 3. Prevent SPA fallback from intercepting reserved API endpoints and docs
    # If the path starts with an API prefix and is NOT an exact frontend page route, return 404
    if norm_path not in EXACT_FRONTEND_PAGE_ROUTES:
        for prefix in RESERVED_PREFIXES:
            if norm_path == prefix or norm_path.startswith(prefix + "/"):
                raise HTTPException(status_code=404, detail="Endpoint not found.")

    # 4. SPA Client-side Route Fallback: return index.html for valid navigation
    index_file = config.FRONTEND_DIST_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(str(index_file))

    # If frontend build is missing in development
    raise HTTPException(
        status_code=503,
        detail="Frontend production build not found. Please run 'npm run build' inside frontend/ to enable production serving.",
    )

