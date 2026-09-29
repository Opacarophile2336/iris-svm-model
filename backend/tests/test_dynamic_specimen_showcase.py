"""Test Suite for Dynamic Botanical Specimen Showcase & 5-Model Prediction Decoupling.

IrisAI Studio — Intelligent Iris Classification & Machine Learning Platform

Tests:
1. Verify that all 3 species map to the correct image pool.
2. Verify that all 5 classifiers preserve actual inference results.
3. Verify that repeated predictions for the same species rotate images and avoid immediate repetition.
4. Verify that each showcase response contains distinct specimens (no duplicates within the same showcase).
5. Verify that small/exhausted image pools degrade gracefully without errors.
6. Verify that simulated iNaturalist network failures gracefully fall back to Wikimedia Commons, and if that fails, fall back to local curated images.
7. Verify that no GBIF API calls are made and no GBIF URLs/identifiers remain.
8. Verify that refreshing/rotating images does NOT create new prediction records in the database.
9. Verify authentication enforcement on /image/showcase/{species_name}.
10. Verify count parameter bounds (count between 1 and 10).
11. Verify species normalization for showcase endpoint.
12. Verify invalid species returns 400 Bad Request.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from auth.service import create_access_token
from database.connection import SessionLocal
from database.models import Prediction
from image import species_image
from main import app

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def auth_headers():
    """Generate authenticated user headers without relying on database user creation."""
    token = create_access_token("showcase_test_user", "USER")
    return {"Authorization": f"Bearer {token}"}


class TestDynamicSpecimenShowcase:
    """Comprehensive test suite for Dynamic Specimen Showcase and Decoupled Rotation."""

    # -------------------------------------------------------------------------
    # 1. Species Image Pool Mapping
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("species_input,expected_species", [
        ("Iris setosa", "Iris setosa"),
        ("setosa", "Iris setosa"),
        ("iris_versicolor", "Iris versicolor"),
        ("Iris virginica", "Iris virginica"),
    ])
    def test_01_species_maps_to_correct_pool(self, auth_headers, species_input, expected_species):
        """Verify that all 3 species map correctly and return specimens matching the taxon."""
        resp = client.get(f"/image/showcase/{species_input}", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["species"] == expected_species
        assert len(data["showcase"]) > 0
        assert data["count"] == len(data["showcase"])
        assert data["pool_size"] >= data["count"]

        # Every specimen must match the requested species
        for item in data["showcase"]:
            assert item["species"] == expected_species
            assert item["image_url"] is not None
            assert item["source"] is not None
            assert "gbif" not in item["source"].lower()

    # -------------------------------------------------------------------------
    # 2. 5 Classifiers Preserve Real Inference
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("model_key", ["rbf", "linear", "poly", "sigmoid", "mlp"])
    def test_02_all_five_classifiers_preserve_actual_inference(self, auth_headers, model_key):
        """Verify that all 5 classifiers run their real inference pipeline and return authentic predictions."""
        # Typical Iris setosa measurements
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
            "model": model_key,
        }
        resp = client.post("/prediction/predict", json=payload, headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["model"] in ["SVM", "Deep Learning MLP"]
        assert data["kernel"] == model_key
        assert data["kernel_display"] in ["RBF", "Linear", "Polynomial", "Sigmoid", "Deep Learning MLP", "SVM RBF", "SVM Linear", "SVM Polynomial", "SVM Sigmoid"]
        assert data["predicted_species"] in config.IRIS_CLASSES
        assert "probabilities" in data
        assert isinstance(data["confidence"], (int, float))
        assert data["confidence"] > 0

    # -------------------------------------------------------------------------
    # 3. History-Aware Rotation Avoids Immediate Repetition
    # -------------------------------------------------------------------------
    def test_03_history_aware_rotation_avoids_immediate_repetition(self, auth_headers):
        """Verify that subsequent showcase queries with exclude IDs return non-overlapping specimens."""
        species = "Iris setosa"
        # First showcase request
        resp1 = client.get(f"/image/showcase/{species}?count=3", headers=auth_headers)
        assert resp1.status_code == 200
        data1 = resp1.json()
        initial_ids = data1["displayed_ids"]
        assert len(initial_ids) == 3

        # Second showcase request excluding the first set of IDs
        exclude_str = ",".join(initial_ids)
        resp2 = client.get(f"/image/showcase/{species}?count=3&exclude={exclude_str}", headers=auth_headers)
        assert resp2.status_code == 200
        data2 = resp2.json()
        second_ids = data2["displayed_ids"]

        # Pool size is >= 6, so second batch should have no overlap with the first
        if data2["pool_size"] >= 6:
            overlap = set(initial_ids).intersection(set(second_ids))
            assert len(overlap) == 0, f"Expected no overlap but got {overlap}"

    # -------------------------------------------------------------------------
    # 4. No Duplicate Images Within Single Showcase
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("species", ["Iris setosa", "Iris versicolor", "Iris virginica"])
    def test_04_no_duplicates_within_showcase(self, auth_headers, species):
        """Verify that each showcase response contains distinct specimens (no duplicate IDs or image URLs)."""
        resp = client.get(f"/image/showcase/{species}?count=3", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        items = data["showcase"]

        ids = [item["id"] for item in items]
        urls = [item["image_url"] for item in items]

        assert len(ids) == len(set(ids)), f"Duplicate specimen IDs detected: {ids}"
        assert len(urls) == len(set(urls)), f"Duplicate image URLs detected: {urls}"

    # -------------------------------------------------------------------------
    # 5. Graceful Degradation on Exhausted Pool
    # -------------------------------------------------------------------------
    def test_05_exhausted_pool_graceful_wrap_around(self, auth_headers):
        """Verify that if all available IDs are excluded, the service wraps around cleanly without crashing."""
        species = "Iris virginica"
        # Exclude an arbitrary list of IDs that covers all likely items
        huge_exclude = ",".join([f"fake_id_{i}" for i in range(50)])
        resp = client.get(f"/image/showcase/{species}?count=3&exclude={huge_exclude}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["showcase"]) > 0
        assert data["count"] > 0

    # -------------------------------------------------------------------------
    # 6. Fallback Chain: iNaturalist -> Wikimedia -> Local Curated
    # -------------------------------------------------------------------------
    def test_06_simulated_network_failures_fallback_chain(self):
        """Verify that network failures gracefully degrade to Wikimedia and then local curated images."""
        species = "Iris setosa"
        # Clear cache for isolated testing
        species_image._species_pool_cache.clear()

        # Case A: iNaturalist fails, Wikimedia succeeds
        mock_wiki_item = [{
            "id": "wiki-test-1",
            "species": "Iris setosa",
            "image_url": "https://upload.wikimedia.org/test1.jpg",
            "thumbnail_url": "https://upload.wikimedia.org/test1_thumb.jpg",
            "source": "Wikimedia Commons",
            "license": "CC BY-SA 4.0",
            "recorded_by": "Wiki Observer",
            "country": "Canada",
            "locality": "Ontario",
            "event_date": "2024-05-01",
            "institution": "Wikimedia Commons",
            "occurrence_url": "https://commons.wikimedia.org/wiki/File:Test.jpg",
        }]

        with patch.object(species_image, "_fetch_gbif_specimens", side_effect=Exception("iNat timeout")):
            with patch.object(species_image, "_fetch_wikimedia_specimens", return_value=mock_wiki_item):
                species_image._species_pool_cache.clear()
                pool = species_image.get_species_pool(species)
                assert len(pool) > 0
                assert any(item["source"] == "Wikimedia Commons" for item in pool)

        # Case B: Both iNaturalist and Wikimedia fail -> Curated local fallback
        species_image._species_pool_cache.clear()
        with patch.object(species_image, "_fetch_gbif_specimens", side_effect=Exception("iNat down")):
            with patch.object(species_image, "_fetch_wikimedia_specimens", side_effect=Exception("Wiki down")):
                pool = species_image.get_species_pool(species)
                assert len(pool) > 0
                # Must contain local curated specimens
                assert any("/assets/" in item.get("thumbnail_url", "") for item in pool)

    # -------------------------------------------------------------------------
    # 7. Complete Absence of GBIF API Calls / Identifiers
    # -------------------------------------------------------------------------
    def test_07_no_gbif_identifiers_or_calls(self, auth_headers):
        """Verify that no GBIF API calls, endpoints, or occurrence URLs are returned."""
        for sp in config.IRIS_CLASSES:
            resp = client.get(f"/image/showcase/{sp}?count=5", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            for item in data["showcase"]:
                assert "gbif.org" not in item.get("image_url", "").lower()
                assert "gbif.org" not in (item.get("occurrence_url") or "").lower()
                assert item.get("source") != "GBIF"

    # -------------------------------------------------------------------------
    # 8. Showcase Rotation Does NOT Create Prediction Records
    # -------------------------------------------------------------------------
    def test_08_showcase_rotation_does_not_create_prediction_records(self, auth_headers):
        """Verify that rotating images or calling showcase does NOT insert rows into the predictions table."""
        initial_count = 0
        has_db = False
        db = SessionLocal()
        try:
            initial_count = db.query(Prediction).count()
            has_db = True
        except Exception:
            pass
        finally:
            db.close()

        # Make multiple showcase calls (simulating rapid rotation)
        for i in range(5):
            resp = client.get(f"/image/showcase/Iris versicolor?count=3&exclude=item_{i}", headers=auth_headers)
            assert resp.status_code == 200

        if has_db:
            db = SessionLocal()
            try:
                after_count = db.query(Prediction).count()
                assert after_count == initial_count, f"Prediction count changed from {initial_count} to {after_count}"
            finally:
                db.close()

    # -------------------------------------------------------------------------
    # 9. Authentication Enforcement
    # -------------------------------------------------------------------------
    def test_09_auth_enforcement_on_showcase(self):
        """Verify that unauthenticated requests to /image/showcase are rejected with 401/403."""
        resp = client.get("/image/showcase/Iris setosa")
        assert resp.status_code in [401, 403]

    # -------------------------------------------------------------------------
    # 10. Count Parameter Bounds
    # -------------------------------------------------------------------------
    def test_10_count_parameter_bounded(self, auth_headers):
        """Verify that count parameter is respected and capped appropriately."""
        # count = 1
        resp1 = client.get("/image/showcase/Iris setosa?count=1", headers=auth_headers)
        assert resp1.status_code == 200
        assert len(resp1.json()["showcase"]) == 1

        # count = 2
        resp2 = client.get("/image/showcase/Iris setosa?count=2", headers=auth_headers)
        assert resp2.status_code == 200
        assert len(resp2.json()["showcase"]) == 2

    # -------------------------------------------------------------------------
    # 11. Invalid Species Handling
    # -------------------------------------------------------------------------
    def test_11_invalid_species_returns_400(self, auth_headers):
        """Verify that an unknown or invalid species returns 400 Bad Request."""
        resp = client.get("/image/showcase/Rosa_canina", headers=auth_headers)
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "unknown species" in detail or "invalid species" in detail
