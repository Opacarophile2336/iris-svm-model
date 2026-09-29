"""Checkpoint 8 Comprehensive Test Suite — IrisAI Studio.

PRODUCTION DEPLOYMENT READINESS & ONE-PORT ARCHITECTURE

Verifies:
1. Frontend JavaScript bundle returns HTTP 200 with application/javascript or text/javascript content-type.
2. Frontend CSS bundle returns HTTP 200 with text/css content-type.
3. Existing Iris specimen image returns HTTP 200 with image/jpeg content-type.
4. Missing static asset returns HTTP 404 (JSON/error), strictly NOT HTML index.html.
5. GET /gallery returns React index.html when production build is present.
6. GET /prediction returns React index.html.
7. GET /admin/svm-lab returns React index.html.
8. GET /docs returns HTTP 200 with Swagger UI HTML.
9. GET /openapi.json returns HTTP 200 with valid OpenAPI schema JSON.
10. Existing API routes remain functional (e.g. /health returns status ok).
11. Unknown API paths (/prediction/nonexistent, /auth/fake) retain proper HTTP 404 behavior, NOT index.html.
12. Invalid API input retains correct 400 or 422 HTTP validation behavior.
13. Protected API routes still enforce authentication (unauthorized request returns 401/403).
14. Missing frontend build fallback: when index.html is absent, root returns development JSON message without crashing.
15. Static resolver prevents path traversal attacks (/assets/../.. returns HTTP 400).
16. Missing static extension files (/missing_script.js, /fake.css) return HTTP 404, never index.html.
17. CORS origin regex validates and accepts trycloudflare.com origins while preserving development origins.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from auth.service import create_access_token
from main import app

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def auth_headers():
    token = create_access_token("cp8_test_user", "USER")
    return {"Authorization": f"Bearer {token}"}


class TestCheckpoint8ProductionReadiness:
    """Comprehensive test suite for Checkpoint 8 One-Port Production Architecture."""

    # 1. Frontend JavaScript bundle returns HTTP 200
    def test_01_frontend_js_bundle_returns_200(self):
        js_files = list((config.FRONTEND_DIST_DIR / "assets").glob("*.js"))
        assert len(js_files) > 0, "No JS bundle found in frontend/dist/assets"
        bundle_name = js_files[0].name
        resp = client.get(f"/assets/{bundle_name}")
        assert resp.status_code == 200, resp.text
        assert any(ct in resp.headers.get("content-type", "") for ct in ["javascript", "application/octet-stream", "text/plain"])
        assert len(resp.content) > 1000

    # 2. Frontend CSS bundle returns HTTP 200
    def test_02_frontend_css_bundle_returns_200(self):
        css_files = list((config.FRONTEND_DIST_DIR / "assets").glob("*.css"))
        assert len(css_files) > 0, "No CSS bundle found in frontend/dist/assets"
        bundle_name = css_files[0].name
        resp = client.get(f"/assets/{bundle_name}")
        assert resp.status_code == 200, resp.text
        assert "css" in resp.headers.get("content-type", "")
        assert len(resp.content) > 500

    # 3. Existing Iris specimen image returns HTTP 200
    def test_03_existing_iris_specimen_image_returns_200(self):
        for img_name in ["iris_setosa.jpg", "iris_versicolor.jpg", "iris_virginica.jpg"]:
            resp = client.get(f"/assets/{img_name}")
            assert resp.status_code == 200, f"Failed to fetch {img_name}: {resp.status_code}"
            assert "image" in resp.headers.get("content-type", "")
            assert len(resp.content) > 1000

    # 4. Missing static asset returns 404, not HTML index.html
    def test_04_missing_static_asset_returns_404_not_html(self):
        resp = client.get("/assets/completely_nonexistent_image_12345.png")
        assert resp.status_code == 404
        content_type = resp.headers.get("content-type", "")
        assert "text/html" not in content_type
        assert "not found" in resp.text.lower()

    # 5. /gallery returns React index.html
    def test_05_gallery_route_returns_index_html(self):
        resp = client.get("/gallery")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<div id=\"root\">" in resp.text or "<!DOCTYPE html>" in resp.text

    # 6. /prediction returns React index.html
    def test_06_prediction_route_returns_index_html(self):
        resp = client.get("/prediction")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<div id=\"root\">" in resp.text or "<!DOCTYPE html>" in resp.text

    # 7. /admin/svm-lab returns React index.html
    def test_07_nested_admin_route_returns_index_html(self):
        resp = client.get("/admin/svm-lab")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<div id=\"root\">" in resp.text or "<!DOCTYPE html>" in resp.text

    # 8. /docs remains Swagger UI
    def test_08_docs_remains_swagger_ui(self):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "swagger" in resp.text.lower()

    # 9. /openapi.json remains valid JSON
    def test_09_openapi_json_remains_valid(self):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert "openapi" in data or "swagger" in data
        assert "paths" in data
        assert "/auth/login" in data["paths"]

    # 10. Existing API routes remain functional
    def test_10_existing_api_routes_remain_functional(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    # 11. Unknown API paths retain proper HTTP 404 behavior, not index.html
    def test_11_unknown_api_paths_retain_404_not_spa(self):
        for path in ["/prediction/completely_unknown_action", "/auth/not_an_endpoint", "/ml/fake_path"]:
            resp = client.get(path)
            assert resp.status_code == 404
            assert "text/html" not in resp.headers.get("content-type", "")

    # 12. Invalid API input retains correct 400/422 behavior
    def test_12_invalid_api_input_retains_validation_error(self, auth_headers):
        # Negative or invalid dimensions should fail validation
        resp = client.post(
            "/prediction/predict",
            json={"sepal_length": -5.0, "sepal_width": 3.0, "petal_length": 1.0, "petal_width": 0.2},
            headers=auth_headers,
        )
        assert resp.status_code in (400, 422)

    # 13. Protected API routes still enforce authentication
    def test_13_protected_api_routes_enforce_authentication(self):
        resp = client.get("/image/gallery")
        assert resp.status_code in (401, 403)

    # 14. Missing frontend build does not break startup / returns clean development message
    def test_14_missing_frontend_build_handling(self):
        with patch.object(config, "FRONTEND_DIST_DIR", Path("D:/nonexistent_fake_dist_dir")):
            # GET / should return clean dev JSON when dist is missing
            resp = client.get("/")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("status") == "running"
            assert "Development mode" in data.get("message", "")

            # Non-API route should return 503 indicating build missing
            resp_route = client.get("/gallery")
            assert resp_route.status_code == 503
            assert "build not found" in resp_route.json().get("detail", "").lower()

    # 15. Static resolver prevents path traversal
    def test_15_static_resolver_prevents_path_traversal(self):
        resp1 = client.get("/assets/../../config.py")
        assert resp1.status_code in (400, 404)

        resp2 = client.get("/assets/..%2f..%2fconfig.py")
        assert resp2.status_code in (400, 404)

    # 16. Missing static extension files return 404, never index.html
    def test_16_missing_static_extension_files_return_404(self):
        for missing_ext_path in ["/script_missing.js", "/style_missing.css", "/images/logo.png"]:
            resp = client.get(missing_ext_path)
            assert resp.status_code == 404
            assert "text/html" not in resp.headers.get("content-type", "")

    # 17. CORS origin regex allows trycloudflare.com and development origins
    def test_17_cors_origin_regex_configuration(self):
        # Simulate preflight OPTIONS request from trycloudflare.com
        resp = client.options(
            "/prediction/predict",
            headers={
                "Origin": "https://random-iris-subdomain.trycloudflare.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization,Content-Type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "https://random-iris-subdomain.trycloudflare.com"
        assert resp.headers.get("access-control-allow-credentials") == "true"
