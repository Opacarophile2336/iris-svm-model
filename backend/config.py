"""Central configuration — reads from .env, never exposed to frontend."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# JWT
JWT_SECRET: str = os.getenv("JWT_SECRET", "hmnc-pro-fallback-secret")
JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRY_HOURS: int = int(os.getenv("JWT_EXPIRY_HOURS", "24"))

# Admin credentials — NEVER sent to frontend
ADMIN_USERNAME: str = os.getenv("ADMIN_USERNAME", "")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

# SQL Server database
DB_DRIVER: str = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
DB_SERVER: str = os.getenv("DB_SERVER", r"VICTUS2336\SQLEXPRESS01")
DB_NAME: str = os.getenv("DB_NAME", "HMNC_PRO")
DB_USER: str = os.getenv("DB_USER", "")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
DB_PORT: str = os.getenv("DB_PORT", "1433")
DB_TRUSTED_CONNECTION: str = os.getenv("DB_TRUSTED_CONNECTION", "yes")
DB_TRUST_SERVER_CERTIFICATE: str = os.getenv(
    "DB_TRUST_SERVER_CERTIFICATE",
    "yes",
)
DB_ENCRYPT: str = os.getenv("DB_ENCRYPT", "yes" if os.getenv("DB_USER") else "no")

# Paths
PROJECT_ROOT = BASE_DIR.parent
USER_ACCOUNTS_PATH = PROJECT_ROOT / "user_accounts.xlsx"
PREDICTION_HISTORY_PATH = PROJECT_ROOT / "data" / "prediction_history.json"
MODEL_HISTORY_PATH = PROJECT_ROOT / "data" / "model_history.json"
TRAINED_MODEL_PATH = PROJECT_ROOT / "data" / "best_model.joblib"
ASSETS_DIR = PROJECT_ROOT / "assets" / "images"
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
PRODUCTION_PORT = int(os.getenv("PORT", "2336"))

# Supported Classifiers for Checkpoint 1 (4 SVM kernels + Deep Learning MLP)
ALLOWED_KERNELS = ["rbf", "linear", "poly", "sigmoid"]
CLASSIFIER_MODELS = ["rbf", "linear", "poly", "sigmoid", "mlp"]

# Iris classes
IRIS_CLASSES = ["Iris setosa", "Iris versicolor", "Iris virginica"]
IRIS_CLASS_KEYS = ["setosa", "versicolor", "virginica"]

