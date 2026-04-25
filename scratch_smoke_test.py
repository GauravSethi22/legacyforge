"""
Smoke test — verifies the full reward system in LegacyforgeEnvironment.

Simulates a well-behaved agent episode:
  1. read_docs  → strategy bonus (no direct reward)
  2. edit_function → patches Flask code to FastAPI
  3. run_tests  → earns real rewards + unlocks Phase 2
  4. submit_test → Triangle of Truth full validation → reward +3 + done=True
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.legacyforge_env import LegacyforgeEnvironment

try:
    from legacyforge.models import LegacyforgeAction
except ImportError:
    from models import LegacyforgeAction

env = LegacyforgeEnvironment()

# ── Reset ────────────────────────────────────────────────────────────────────
obs = env.reset()
print("=" * 60)
print("RESET")
print(f"  phase={obs.info['phase']}  level={obs.level}")

# ── Step 1: read_docs ────────────────────────────────────────────────────────
obs = env.step(LegacyforgeAction(action_type="read_docs", target="async"))
print("\nStep 1 — read_docs(async)")
print(f"  reward={obs.reward}  summary={obs.migration_history_summary[:60]}")

# ── Step 2: edit_function — inject correct FastAPI code ──────────────────────
FASTAPI_CODE = """\
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ItemResponse(BaseModel):
    item_id: int
    name: str

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int):
    if item_id > 0:
        return ItemResponse(item_id=item_id, name="Test Item")
    raise HTTPException(status_code=404, detail="Item not found")
"""

obs = env.step(LegacyforgeAction(action_type="edit_function",
                                  target="read_item",
                                  code=FASTAPI_CODE))
print("\nStep 2 — edit_function(read_item)")
print(f"  reward={obs.reward}  compile_ok={obs.info.get('compile_ok')}")

# ── Step 3: run_tests ────────────────────────────────────────────────────────
obs = env.step(LegacyforgeAction(action_type="run_tests"))
print("\nStep 3 — run_tests()")
print(f"  reward={obs.reward}")
print(f"  test_passage_rate={obs.info.get('test_passage_rate')}")
print(f"  phase after={obs.info.get('phase')}")
print(f"  reward_breakdown={obs.reward_breakdown}")

assert obs.info.get("phase") == 2, "Phase 2 should be unlocked after 100% test passage!"

# ── Step 4: submit_test — adversarial test that only passes on good code ─────
ADVERSARIAL_TEST = """\
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)

def test_route_exists_and_200():
    response = client.get("/items/1")
    assert response.status_code == 200

def test_response_json_shape():
    response = client.get("/items/1")
    data = response.json()
    assert "item_id" in data
    assert "name" in data

def test_invalid_item_id_404():
    response = client.get("/items/0")
    assert response.status_code == 404
"""

obs = env.step(LegacyforgeAction(action_type="submit_test", code=ADVERSARIAL_TEST))
print("\nStep 4 — submit_test(adversarial_test)")
print(f"  reward={obs.reward}")
print(f"  done={obs.done}")
print(f"  reason={obs.info.get('validation', {}).get('reason')}")
print(f"  reward_breakdown={obs.reward_breakdown}")

# ── Final state ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL STATE")
state = env.state
for k, v in state.items():
    if k != "episode_log":
        print(f"  {k}: {v}")
print(f"  episode_log entries: {len(state['episode_log'])}")

print("\nAll smoke test steps PASSED!")
