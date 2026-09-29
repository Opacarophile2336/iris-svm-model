# HMNC_PRO — Machine Learning & Deep Learning Platform

HMNC_PRO is a comprehensive Machine Learning and Deep Learning system built with a **FastAPI** backend and a **React + TypeScript + Vite** frontend. It features role-based access control (USER / ADMIN), Excel-based account persistence, multi-kernel Support Vector Machine (SVM) training and comparative inference with cross-validation (RBF, Linear, Polynomial — SIGMOID strictly excluded), batch dataset processing (.csv, .txt, .xlsx), and an interactive Iris Prediction Studio integrated with real botanical specimen photograph retrieval (GBIF & Wikimedia Commons).

---

## 1. System Architecture

```
HMNC_PRO
│
├── backend/
│   ├── main.py                  # FastAPI entry point & CORS configuration
│   ├── config.py                # Environment configuration & constants
│   ├── .env                     # Secure admin credentials & JWT secret
│   ├── auth/                    # Registration, authentication & role dependencies
│   ├── storage/                 # Excel account repository (openpyxl)
│   ├── ml/                      # SVM training, cross-validation, decision boundaries, history
│   ├── dl/                      # Deep Learning pipeline skeleton (Data → Preprocessing → Model → Training → Evaluation → Prediction)
│   ├── prediction/              # Iris prediction studio API, batch upload/export & insights
│   │   ├── router.py            # Endpoints: /predict, /predict-all, /batch-upload, /batch-export, /insights, /metrics
│   │   ├── batch_service.py     # Safe CSV/TXT/XLSX parser & openpyxl report generator
│   │   ├── insights_service.py  # Z-score & botanical morphological diagnostics engine
│   │   └── history_repository.py# Per-user prediction history
│   ├── datasets/                # Iris dataset explorer, quality stats, feature dictionary
│   ├── image/                   # Species image service (GBIF → Wikimedia → verified local fallback)
│   └── tests/                   # Automated pytest suite (38 test scenarios)
│
├── frontend/
│   ├── src/
│   │   ├── api/                 # Axios clients with JWT bearer interceptor
│   │   ├── auth/                # AuthContext & useAuth hook
│   │   ├── routes/              # ProtectedRoute (role-aware navigation guard)
│   │   ├── components/          # Layout, role-based navigation sidebar
│   │   ├── pages/
│   │   │   ├── auth/            # Simple Login & Register pages
│   │   │   ├── user/            # Dashboard, Datasets, Quality, Dictionary, PredictionStudio, PredictionHistory, ModelInsights, ModelHub
│   │   │   └── admin/           # SVMModelLab, Evaluation, DecisionBoundary, ModelHistory
│   │   └── index.css            # Clean, professional, restrained styling
│   ├── vite.config.ts           # Configured for Frontend port 2336 & Backend proxy 2006
│   └── package.json
│
├── assets/images/               # Local verified real specimen photographs (GBIF offline backup)
├── user_accounts.xlsx           # Excel user account storage (auto-created if missing)
└── run_app.py                   # One-click launcher for backend & frontend
```

---

## 2. Ports & Network Configuration

- **Frontend Application**: `http://localhost:2336`
- **Backend API**: `http://127.0.0.1:2006`
- **Interactive API Documentation (Swagger)**: `http://127.0.0.1:2006/docs`

---

## 3. Authentication & Role Permissions

The system implements strict, independent role-based access control (RBAC) enforced on both frontend and backend.

### Two Account Types
1. **USER**:
   - Created via the public Register page (`/register`).
   - Registration inputs: `Username`, `Password`, `Confirm Password`.
   - Access to: Dashboard, Datasets, Quality, Dictionary, Prediction Studio, Prediction History, Model Insights, Model Hub.
   - Isolated history: Each user only sees their own prediction records.

2. **ADMIN**:
   - Fixed, pre-configured administrative account.
   - Credentials configured in `.env` (never exposed in frontend code or user Excel table):
     - **Username**: `0814230306`
     - **Password**: `0814230306`
   - Access to: All USER features plus **SVM Model Lab**, **Evaluation**, **Decision Boundary**, and **Model History**.

### Excel Storage (`user_accounts.xlsx`)
- Stored at project root: `E:\8_Project 2026\HMNC_PRO\user_accounts.xlsx`.
- Sheet: `Users`.
- Columns: `username`, `password_hash`, `created_at`, `last_login`, `role`, `status`.
- Passwords are securely hashed with `bcrypt`. Plaintext passwords are never stored.
- Existing records are strictly preserved and never overwritten on startup.

---

## 4. Machine Learning & SVM Configuration

### Supported Kernels
- **RBF** (Radial Basis Function)
- **LINEAR**
- **POLY** (Polynomial, degree=3)
- ⚠️ **SIGMOID is completely removed** from all workflows, APIs, forms, and decision boundary generators.

### 3-Kernel Comparative Inference
- When evaluating samples, predictions are generated across **all three kernels** simultaneously.
- Real validated test accuracy (e.g. 96.67% RBF, 100% Linear, 90% Poly) and 5-fold cross-validation scores from actual model evaluations are reported.
- Automatic consensus analysis calculates whether models reach **Unanimous** (3/3), **Majority** (2/3), or **Divergent** decisions.

---

## 5. Prediction Studio & Batch Dataset Upload

### A. Manual Single Prediction
- Inputs: `Sepal Length`, `Sepal Width`, `Petal Length`, `Petal Width` (cm).
- Outputs side-by-side comparison cards for RBF, Linear, and Polynomial models.
- Displays peak confidence, probability breakdowns across all 3 classes, and the decision consensus.

### B. Batch Dataset Processing (.csv, .txt, .xlsx)
- Users can upload batch files up to 10 MB (up to 5,000 samples).
- **Auto-normalization**: Automatically maps common column variations (e.g. `sepal_length`, `Sepal Length (cm)`, `sl`).
- Rejects files with missing features or malformed numbers gracefully with descriptive error feedback.
- Predicts every sample row across RBF, Linear, and Poly models.

### C. Downloadable Reports
- **XLSX Report** (Primary format):
  - Sheet 1: `Batch Predictions` (Row #, 4 Features, RBF/Linear/Poly Predictions & Confidences, Consensus)
  - Sheet 2: `Model Evaluation Summary` (Real Test Accuracy %, CV Accuracy %, Precision %, Recall %, F1-Score %)
  - Formatted with styled header fills, borders, and auto-adjusted column widths via `openpyxl`.
- **CSV Report**: Clean tabular export of batch results.

### D. Real Botanical Specimen Photograph Retrieval
- Strict authentic photograph requirement: **No AI, synthetic drawings, or SVG artwork**.
- Image retrieval cascade:
  1. Primary: **GBIF API** (Global Biodiversity Information Facility) media search.
  2. Secondary: **Wikimedia Commons** direct media lookup.
  3. Offline Fallback: High-resolution real specimen photographs stored under `assets/images/`.
  4. Non-available Fallback: Textual banner *"Real specimen image currently unavailable."*
  5. Displays authentic source attribution (e.g. `Image Source: GBIF / Wikimedia Commons`).

---

## 6. Innovative Sidebar Additions

To provide genuine scientific value without visual clutter, two high-value modules were integrated:

1. **💡 Model Insights (`/insights`)**:
   - Compares sample dimensions against Fisher's classic 150-sample Iris empirical distributions.
   - Computes morphological Z-scores and standardized distance index for Setosa, Versicolor, and Virginica.
   - Provides academic decision boundary diagnostics explaining why the models classified the specimen (e.g., petal length as linear separator).
   - Flags out-of-distribution botanical anomalies and outliers.

2. **🔬 Model Benchmark & Kernel Hub (`/benchmark`)**:
   - Side-by-side comparison of the 3 kernels with mathematical formulations:
     - RBF: $K(x, x') = \exp(-\gamma ||x - x'||^2)$
     - Linear: $K(x, x') = x^T x'$
     - Polynomial: $K(x, x') = (\gamma x^T x' + r)^d$
   - Real validated metrics table from the hold-out test set.
   - Live system runtime monitors (Backend Port 2006 status, Frontend Port 2336 status, optimal CV kernel).

---

## 7. How to Run the Application

### Option A: Using the Launcher (Recommended)
Run from the project root:
```bash
python run_app.py
```
This will:
1. Verify system dependencies and node environment.
2. Initialize `user_accounts.xlsx` if missing.
3. Automatically load or train all three SVM models.
4. Launch the FastAPI backend on `http://127.0.0.1:2006`.
5. Launch the Vite frontend on `http://localhost:2336`.
6. Open your default web browser to the application.

### Option B: Starting Manually

**Terminal 1 — Backend:**
```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 2006 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

---

## 8. Running Automated Tests

Run the full automated test suite (38 comprehensive test scenarios):
```bash
cd backend
python -m pytest tests/test_api.py -v
```

Tests cover:
- USER registration validation, duplicate handling, password confirmation.
- USER & ADMIN authentication with JWT generation.
- Role-based authorization and rejection of unauthorized access (401/403).
- Model prediction accuracy, probabilities sum, and feature bounds.
- Kernel restriction (ensuring SIGMOID is rejected).
- Per-user prediction history data isolation.
- Multi-kernel prediction across RBF, Linear, and Poly.
- Batch file uploads for CSV, TXT, and XLSX formats.
- Invalid file rejection and graceful error handling.
- Downloadable XLSX and CSV batch report generation.
- Model insights generation and statistical diagnostics.
- Species image authentic source resolution.
