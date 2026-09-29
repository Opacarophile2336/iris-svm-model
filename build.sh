#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "=== 1. Installing Python dependencies ==="
pip install --upgrade pip
pip install -r backend/requirements.txt

echo "=== 2. Building React frontend ==="
cd frontend
npm install
npm run build
cd ..

echo "=== 3. Build completed successfully ==="
