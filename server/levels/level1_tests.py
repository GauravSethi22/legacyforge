import pytest
from fastapi.testclient import TestClient

# The sandbox will dynamically inject 'app_module'
try:
    from app_module import app  # type: ignore  # injected dynamically by sandbox.py
except ImportError:
    from server.levels.level1_answer import app  # type: ignore

client = TestClient(app)

def test_route_exists_and_200():
    response = client.get("/items/1")
    assert response.status_code == 200

def test_response_json_shape():
    response = client.get("/items/1")
    data = response.json()
    assert "item_id" in data
    assert "name" in data

def test_name_is_item_specific():
    """Name must be unique per item_id, not hardcoded."""
    r1 = client.get("/items/1").json()
    r2 = client.get("/items/2").json()
    assert r1["name"] != r2["name"], "name must differ per item_id"

def test_zero_returns_422():
    """item_id=0 is non-positive and must return 422."""
    response = client.get("/items/0")
    assert response.status_code == 422

def test_negative_returns_422():
    """Negative item_ids must return 422."""
    response = client.get("/items/-5")
    assert response.status_code == 422

def test_out_of_range_returns_404():
    """item_id > 1000 must return 404."""
    response = client.get("/items/9999")
    assert response.status_code == 404
