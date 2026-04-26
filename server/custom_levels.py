"""
custom_levels.py
A dataset of dynamic legacy code migration levels for LegacyForge RL training.
"""

my_custom_levels = [

    # ---------------------------------------------------------
    # LEVEL 1: The Original Default Level
    # ---------------------------------------------------------
    {
        "level_name": "Level 1: Basic Routing",
        "test_suite_size": 6,
        "flask_code": None,   # Triggers fallback to level1_flask.py
        "test_code": None,    # Triggers fallback to level1_tests.py
        "golden_code": None   # Triggers fallback to golden_solution_l1.py
    },

    # ---------------------------------------------------------
    # LEVEL 2: POST Requests and Pydantic Payloads
    # ---------------------------------------------------------
    {
        "level_name": "Level 2: CRUD and POST Payloads",
        "test_suite_size": 3,
        "flask_code": """\
from flask import Flask, request, jsonify

app = Flask(__name__)
users_db = {}

@app.route('/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'username' not in data:
        return jsonify({"error": "Missing username"}), 400

    u_id = len(users_db) + 1
    users_db[u_id] = data['username']
    return jsonify({"id": u_id, "username": data['username']}), 201

@app.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    if user_id not in users_db:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": user_id, "username": users_db[user_id]}), 200
""",
        "test_code": """\
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)

def test_create_user():
    response = client.post("/users", json={"username": "Alice"})
    assert response.status_code == 201
    assert response.json() == {"id": 1, "username": "Alice"}

def test_get_user():
    client.post("/users", json={"username": "Bob"})
    response = client.get("/users/2")
    assert response.status_code == 200
    assert response.json() == {"id": 2, "username": "Bob"}

def test_get_missing_user():
    response = client.get("/users/999")
    assert response.status_code == 404
""",
        "golden_code": """\
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()
users_db = {}

class UserCreate(BaseModel):
    username: str

class UserResponse(BaseModel):
    id: int
    username: str

@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    u_id = len(users_db) + 1
    users_db[u_id] = user.username
    return UserResponse(id=u_id, username=user.username)

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    if user_id not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(id=user_id, username=users_db[user_id])
"""
    },

    # ---------------------------------------------------------
    # LEVEL 3: Query Parameters and Default Values
    # ---------------------------------------------------------
    {
        "level_name": "Level 3: Query Parameters",
        "test_suite_size": 3,
        "flask_code": """\
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search_items():
    query = request.args.get('q')
    limit = request.args.get('limit', default=10, type=int)

    if not query:
        return jsonify({"error": "Search query 'q' is required"}), 400

    results = [f"Result for {query}"] * limit
    return jsonify({"query": query, "count": limit, "results": results}), 200
""",
        "test_code": """\
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)

def test_search_with_defaults():
    response = client.get("/search?q=apple")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "apple"
    assert data["count"] == 10
    assert len(data["results"]) == 10

def test_search_with_custom_limit():
    response = client.get("/search?q=banana&limit=2")
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert len(response.json()["results"]) == 2

def test_search_missing_query():
    # FastAPI automatically throws a 422 Unprocessable Entity if a required query param is missing
    response = client.get("/search")
    assert response.status_code in [400, 422]
""",
        "golden_code": """\
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List

app = FastAPI()

class SearchResult(BaseModel):
    query: str
    count: int
    results: List[str]

@app.get("/search", response_model=SearchResult)
async def search_items(q: str = Query(..., min_length=1), limit: int = 10):
    results = [f"Result for {q}"] * limit
    return SearchResult(query=q, count=limit, results=results)
"""
    },

    # ---------------------------------------------------------
    # LEVEL 4: Headers and Custom Status Exceptions
    # ---------------------------------------------------------
    {
        "level_name": "Level 4: Headers and Auth",
        "test_suite_size": 3,
        "flask_code": """\
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/secure-data', methods=['GET'])
def get_secure_data():
    auth_token = request.headers.get('X-Auth-Token')

    if not auth_token:
        return jsonify({"message": "Missing token"}), 401

    if auth_token != "supersecret":
        return jsonify({"message": "Invalid token"}), 403

    return jsonify({"data": "Confidential information"}), 200
""",
        "test_code": """\
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)

def test_secure_data_success():
    response = client.get("/secure-data", headers={"X-Auth-Token": "supersecret"})
    assert response.status_code == 200
    assert response.json() == {"data": "Confidential information"}

def test_secure_data_missing_header():
    response = client.get("/secure-data")
    assert response.status_code in [401, 422]

def test_secure_data_invalid_header():
    response = client.get("/secure-data", headers={"X-Auth-Token": "wrongpassword"})
    assert response.status_code == 403
""",
        "golden_code": """\
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class SecureData(BaseModel):
    data: str

@app.get("/secure-data", response_model=SecureData)
async def get_secure_data(x_auth_token: Optional[str] = Header(None)):
    if not x_auth_token:
        raise HTTPException(status_code=401, detail="Missing token")
    if x_auth_token != "supersecret":
        raise HTTPException(status_code=403, detail="Invalid token")

    return SecureData(data="Confidential information")
"""
    }
]
