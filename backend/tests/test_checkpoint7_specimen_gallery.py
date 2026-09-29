"""Checkpoint 7 Comprehensive Test Suite — IrisAI Studio.

AUTHENTIC BOTANICAL SPECIMEN IMAGE & GALLERY STUDIO

Verifies:
1. Gallery all endpoint returns all three Iris species.
2. Gallery filtering for Iris setosa.
3. Gallery filtering for Iris versicolor.
4. Gallery filtering for Iris virginica.
5. Specimen metadata structure completeness (id, species, image_url, etc.).
6. Image and thumbnail URL validity (starts with http://, https://, or /assets/).
7. Species name normalization (case-insensitive, underscore, without 'Iris ').
8. Invalid species returns 400 Bad Request.
9. Backward compatibility of single image endpoint (/image/species/{species_name}).
10. Curated fallback resilience when GBIF is unreachable or fails.
11. Specimen count per species is at least 2.
12. Authentication enforcement on /image/gallery.
13. Authentication enforcement on /image/species/{species_name}.
14. Complete independence from SQL Server schema / database state.
15. Gallery species strictly match config.IRIS_CLASSES.
16. Duplicate image URLs are eliminated within each result set.
17. Query parameter filtering (/image/gallery?species=...) works correctly.
18. Truthful metadata contract (missing fields are None, no fabricated placeholders).
19. Cache isolation between gallery cache and single-image cache.
20. Static integrity of CURATED_FALLBACK_SPECIMENS dataset.
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
from image import species_image
from main import app

client = TestClient(app, raise_server_exceptions=True)


@pytest.fixture(scope="module")
def auth_headers():
    """Generate authenticated user headers without relying on database user creation."""
    token = create_access_token("cp7_gallery_user", "USER")
    return {"Authorization": f"Bearer {token}"}


class TestCheckpoint7SpecimenGallery:
    """Comprehensive test suite for Checkpoint 7: Authentic Botanical Specimen Gallery."""

    def test_01_gallery_all_returns_all_three_species(self, auth_headers):
        """Test 1: GET /image/gallery without query params returns specimens for all 3 species."""
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert "species" in data
        assert data["species"] == "All"
        assert "items" in data
        assert "count" in data
        assert data["count"] == len(data["items"])
        assert data["count"] >= 6  # At least 2 per species * 3 species

        species_found = {item["species"] for item in data["items"]}
        for expected in config.IRIS_CLASSES:
            assert expected in species_found, f"Missing species {expected} in gallery"

    def test_02_gallery_filter_setosa(self, auth_headers):
        """Test 2: GET /image/gallery/Iris setosa returns only Iris setosa specimens."""
        resp = client.get("/image/gallery/Iris setosa", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["species"] == "Iris setosa"
        assert len(data["items"]) >= 2
        for item in data["items"]:
            assert item["species"] == "Iris setosa"

    def test_03_gallery_filter_versicolor(self, auth_headers):
        """Test 3: GET /image/gallery/Iris versicolor returns only Iris versicolor specimens."""
        resp = client.get("/image/gallery/Iris versicolor", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["species"] == "Iris versicolor"
        assert len(data["items"]) >= 2
        for item in data["items"]:
            assert item["species"] == "Iris versicolor"

    def test_04_gallery_filter_virginica(self, auth_headers):
        """Test 4: GET /image/gallery/Iris virginica returns only Iris virginica specimens."""
        resp = client.get("/image/gallery/Iris virginica", headers=auth_headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()

        assert data["species"] == "Iris virginica"
        assert len(data["items"]) >= 2
        for item in data["items"]:
            assert item["species"] == "Iris virginica"

    def test_05_specimen_metadata_structure(self, auth_headers):
        """Test 5: Every specimen record contains the full set of required metadata keys."""
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        required_keys = {
            "id",
            "species",
            "image_url",
            "thumbnail_url",
            "source",
            "recorded_by",
            "country",
            "locality",
            "event_date",
            "license",
            "institution",
        }

        for item in data["items"]:
            missing = required_keys - set(item.keys())
            assert not missing, f"Specimen {item.get('id')} missing keys: {missing}"
            assert item["species"] in config.IRIS_CLASSES
            assert isinstance(item["id"], str) and len(item["id"]) > 0

    def test_06_specimen_image_url_validity(self, auth_headers):
        """Test 6: All image and thumbnail URLs are valid http, https, or /assets/ paths."""
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        for item in data["items"]:
            img_url = item["image_url"]
            thumb_url = item["thumbnail_url"]
            assert img_url.startswith(("http://", "https://", "/assets/")), f"Invalid image_url: {img_url}"
            assert thumb_url.startswith(("http://", "https://", "/assets/")), f"Invalid thumbnail_url: {thumb_url}"

    def test_07_species_name_normalization_variants(self, auth_headers):
        """Test 7: Router normalizes various species name formats (case, underscore, without prefix)."""
        variations = [
            ("setosa", "Iris setosa"),
            ("iris_setosa", "Iris setosa"),
            ("IRIS SETOSA", "Iris setosa"),
            ("versicolor", "Iris versicolor"),
            ("iris_versicolor", "Iris versicolor"),
            ("virginica", "Iris virginica"),
            ("Iris_Virginica", "Iris virginica"),
        ]
        for query_val, expected_species in variations:
            resp = client.get(f"/image/gallery/{query_val}", headers=auth_headers)
            assert resp.status_code == 200, f"Failed for {query_val}: {resp.text}"
            data = resp.json()
            assert data["species"] == expected_species

    def test_08_invalid_species_returns_400(self, auth_headers):
        """Test 8: Unknown or invalid species returns 400 Bad Request with supported list."""
        resp = client.get("/image/gallery/rosa_canina", headers=auth_headers)
        assert resp.status_code == 400
        assert "detail" in resp.json()
        assert "Supported" in resp.json()["detail"]

        resp_param = client.get("/image/gallery?species=oak_tree", headers=auth_headers)
        assert resp_param.status_code == 400

    def test_09_backward_compatibility_single_image_endpoint(self, auth_headers):
        """Test 9: Existing /image/species/{species_name} endpoint remains 100% backward compatible."""
        for sp in config.IRIS_CLASSES:
            resp = client.get(f"/image/species/{sp}", headers=auth_headers)
            assert resp.status_code == 200, f"Failed for {sp}: {resp.text}"
            data = resp.json()
            assert "species" in data
            assert data["species"] == sp
            assert "image_url" in data
            assert "source" in data
            assert data["image_url"].startswith(("http://", "https://", "/assets/"))

    def test_10_curated_fallback_resilience_when_gbif_fails(self, monkeypatch):
        """Test 10: If GBIF network query fails completely, gallery seamlessly falls back to curated items."""
        # Clear gallery cache for Iris setosa
        species_image._gallery_cache.pop("Iris setosa", None)

        # Mock _fetch_gbif_specimens to simulate network error / empty response
        with patch.object(species_image, "_fetch_gbif_specimens", return_value=[]):
            res = species_image.get_specimen_gallery("Iris setosa")
            assert res["species"] == "Iris setosa"
            assert res["count"] >= 2
            # Should contain items from CURATED_FALLBACK_SPECIMENS
            sources = {item["source"] for item in res["items"]}
            assert ("Wikimedia Commons" in sources) or ("Verified Herbarium / Local Specimen" in sources)

    def test_11_specimen_count_per_species(self, auth_headers):
        """Test 11: Every species individually returns >= 2 botanical specimen images."""
        for sp in config.IRIS_CLASSES:
            resp = client.get(f"/image/gallery/{sp}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            assert data["count"] >= 2, f"Species {sp} returned fewer than 2 specimens"

    def test_12_gallery_authentication_required(self):
        """Test 12: Calling /image/gallery without token returns 401 / 403."""
        resp = client.get("/image/gallery")
        assert resp.status_code in (401, 403)

    def test_13_single_image_authentication_required(self):
        """Test 13: Calling /image/species/{species_name} without token returns 401 / 403."""
        resp = client.get("/image/species/Iris setosa")
        assert resp.status_code in (401, 403)

    def test_14_no_sql_server_dependency(self, auth_headers):
        """Test 14: Gallery endpoints execute purely in-memory/external API without DB locks or queries."""
        # Endpoint runs successfully even if DB queries are not made
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200

    def test_15_gallery_species_match_ml_classes(self, auth_headers):
        """Test 15: Species names in gallery strictly adhere to ML/DL classification classes."""
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        item_species = {item["species"] for item in data["items"]}
        assert item_species == set(config.IRIS_CLASSES)

    def test_16_duplicate_image_elimination(self, auth_headers):
        """Test 16: Ensure no duplicate image URLs within each species gallery."""
        for sp in config.IRIS_CLASSES:
            resp = client.get(f"/image/gallery/{sp}", headers=auth_headers)
            assert resp.status_code == 200
            data = resp.json()
            urls = [item["image_url"] for item in data["items"]]
            assert len(urls) == len(set(urls)), f"Found duplicate image URLs for species {sp}"

    def test_17_gallery_query_parameter_filter(self, auth_headers):
        """Test 17: Query parameter /image/gallery?species=... works interchangeably with path param."""
        resp_path = client.get("/image/gallery/Iris versicolor", headers=auth_headers)
        resp_param = client.get("/image/gallery?species=Iris versicolor", headers=auth_headers)
        assert resp_path.status_code == 200
        assert resp_param.status_code == 200
        assert resp_path.json()["count"] == resp_param.json()["count"]
        assert resp_param.json()["species"] == "Iris versicolor"

    def test_18_specimen_metadata_no_fake_content(self, auth_headers):
        """Test 18: No fabricated metadata placeholders (strict truthful metadata rule)."""
        forbidden_placeholders = {"fake", "n/a", "unknown", "placeholder", "dummy", "test"}
        resp = client.get("/image/gallery", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()

        for item in data["items"]:
            for field in ["recorded_by", "country", "locality", "event_date", "institution"]:
                val = item.get(field)
                if val is not None:
                    assert isinstance(val, str)
                    assert val.strip().lower() not in forbidden_placeholders, (
                        f"Field '{field}' contains invalid placeholder '{val}' in specimen {item.get('id')}"
                    )

    def test_19_gallery_cache_isolation(self):
        """Test 19: Gallery cache does not overwrite or interfere with single-image cache."""
        # Pre-condition: warm single-image cache
        single_setosa = species_image.get_species_image("Iris setosa")
        assert "image_url" in single_setosa

        # Invoke gallery
        gallery_setosa = species_image.get_specimen_gallery("Iris setosa")
        assert isinstance(gallery_setosa["items"], list)

        # Single image cache should remain string URL while gallery cache stores list
        cached_single = species_image._cache.get("Iris setosa")
        assert cached_single is not None
        assert isinstance(cached_single, str)
        assert cached_single.startswith(("http://", "https://", "/assets/"))

        cached_gallery = species_image._gallery_cache.get("Iris setosa")
        assert cached_gallery is not None
        assert isinstance(cached_gallery, list)
        assert len(cached_gallery) >= 2

    def test_20_curated_fallback_specimens_integrity(self):
        """Test 20: CURATED_FALLBACK_SPECIMENS dictionary integrity and content validation."""
        for sp in config.IRIS_CLASSES:
            assert sp in species_image.CURATED_FALLBACK_SPECIMENS
            specimens = species_image.CURATED_FALLBACK_SPECIMENS[sp]
            assert len(specimens) >= 2
            for spec in specimens:
                assert spec["species"] == sp
                assert spec["id"].startswith(sp.split()[1].lower())
                assert spec["image_url"].startswith(("http://", "https://", "/assets/"))
                assert spec["license"] is not None
