"""
generate_golden_data.py
Runs llama-3.3-70b-versatile to generate successful SFT training data.
Saves ONLY episodes that achieve 100% test passage in ShareGPT format.
"""
import json
import os
import sys
import time
import httpx
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

GROQ_TOKEN = os.environ.get("GROQ_API_KEY", "").strip()

from server.legacyforge_env import LegacyforgeEnvironment
from models import LegacyforgeAction

MODEL_ID       = "llama-3.3-70b-versatile"
NUM_EPISODES   = 100  # Set this to whatever you need
MAX_ACTIONS    = 20
OUTPUT_FILE    = os.path.join(_HERE, "sft_dataset.json")

_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_TOKEN}",
    "Content-Type": "application/json",
}

SYSTEM_PROMPT_TEMPLATE = """\
You are migrating a Flask app to FastAPI. Your goal: make ALL tests pass.

THE TESTS EXPECT:
- A module with `app = FastAPI()` that can be imported
- GET /items/{item_id} returns 200 with {"item_id": int, "name": str}
- The `name` MUST be unique per item_id (e.g., f"Item {item_id}")
- item_id <= 0 MUST return 422 (Validation Error)
- item_id > 1000 MUST return 404 (Not Found)
- You MUST use `from fastapi import FastAPI, HTTPException`
- You MUST use async def

TARGET STRUCTURE (this is what passing code looks like):
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class ItemResponse(BaseModel):
    item_id: int
    name: str

@app.get("/items/{item_id}", response_model=ItemResponse)
async def read_item(item_id: int):
    if item_id <= 0:
        raise HTTPException(status_code=422, detail="Item ID must be positive")
    if item_id > 1000:
        raise HTTPException(status_code=404, detail="Item not found")
    return ItemResponse(item_id=item_id, name=f"Item {item_id}")

ACTION ORDER:

Call read_docs

Call edit_function

Call run_tests

If passage < 70%, call code_review to analyze the test failures, THEN call edit_function again.

If passage >= 70%, call submit_test

CRITICAL: For edit_function, new_code must be the ENTIRE module -
all imports, app = FastAPI(), models, and route handlers.
NOT just a single function.

RESPONSE FORMAT (Strict JSON only, no markdown):
{"action": "edit_function", "params": {"name": "module", "new_code": "<INSERT_ENTIRE_MODULE_CODE_HERE_AS_ESCAPED_STRING>"}}
"""

def extract_json(text: str) -> dict:
    text = re.sub(r"json|", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    blocks = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start != -1:
                blocks.append(text[start:i+1])
                start = -1

    last_valid_json = None
    for block in blocks:
        try:
            parsed = json.loads(block)
            last_valid_json = parsed
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    if last_valid_json is not None:
        return last_valid_json

    raise ValueError("No valid JSON object containing an action was found.")

def call_model(user_msg: str) -> dict | None:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    # Simple retry logic for rate limits
    for attempt in range(3):
        try:
            resp = httpx.post(_GROQ_URL, headers=_GROQ_HEADERS, json=payload, timeout=60)
            if resp.status_code == 429:
                print(f"    [!] Rate limited by Groq. Waiting {10 * (attempt+1)}s...")
                time.sleep(10 * (attempt + 1))
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return extract_json(raw)
        except Exception as e:
            if attempt == 2:
                print(f"    [!] API or JSON Error: {e}")
            time.sleep(2)
            
    return None

def model_response_to_action(response: dict) -> LegacyforgeAction | None:
    try:
        action_type = response.get("action", "")
        params      = response.get("params", {}) or {}
        return LegacyforgeAction(
            action_type=action_type,
            target=params.get("topic") or params.get("name") or params.get("justification"),
            code=params.get("new_code") or params.get("test_code"),
        )
    except Exception:
        return None

def run_episode(env: LegacyforgeEnvironment) -> list | None:
    obs = env.reset(level=1)
    action_history = []
    trajectory = []

    for step_idx in range(1, MAX_ACTIONS + 1):
        time.sleep(3) # Groq rate limit buffer
        current_state = env.state
        history_str = " -> ".join(action_history[-5:]) if action_history else "None"
        
        last_action = action_history[-1] if action_history else None
        read_docs_count = current_state["read_docs_count"]

        if last_action == "read_docs" and read_docs_count >= 1:
            force_hint = "You MUST call edit_function now. Do NOT call read_docs again."
        elif last_action == "edit_function":
            force_hint = "You MUST call run_tests now. Do NOT call anything else."
        elif last_action == "run_tests" and current_state['test_passage_rate'] >= 0.70:
            force_hint = "Passage rate is above 70%. You MUST call submit_test now."
        elif last_action == "run_tests" and current_state['test_passage_rate'] < 0.70:
            force_hint = "Tests failing! Call edit_function and provide the ENTIRE FASTAPI MODULE. Do NOT use placeholders or '...'. You must write the complete, functional code!"
        else:
            force_hint = "Start by calling read_docs with topic async or routing."

        user_msg = SYSTEM_PROMPT_TEMPLATE + f"""
Recent actions: {history_str}
read_docs calls this episode: {read_docs_count} / 2 max

HINT: {force_hint}

Current legacy code:
{obs.legacy_code[:800]}

Current passage rate: {current_state['test_passage_rate']:.2f}
Current phase: {current_state['phase']}

Respond with a single JSON object only.
"""

        response = call_model(user_msg)
        if response is None:
            print(f"  Step {step_idx:2d} | ❌ JSON Parse or API Error. Aborting episode.")
            return None 
            
        action = model_response_to_action(response)
        if action is None:
            print(f"  Step {step_idx:2d} | ❌ Invalid Action Format. Aborting episode.")
            return None

        trajectory.append({"from": "user", "value": user_msg})
        trajectory.append({"from": "assistant", "value": json.dumps(response, indent=2)})

        action_history.append(action.action_type)
        obs = env.step(action)
        
        print(f"  Step {step_idx:2d} | Action: {action.action_type:<15} | Passage: {env.state['test_passage_rate']:.2f} | Phase: {env.state['phase']}")

        if obs.done:
            if env.state["test_passage_rate"] == 1.0:
                print(f"  ✅ Perfect run! Tests passed and adversarial test succeeded.")
                return trajectory
            print(f"  ❌ Episode ended, but test passage was only {env.state['test_passage_rate']:.2f}")
            return None
            
    print(f"  ❌ Timed out after {MAX_ACTIONS} steps.")
    return None

def main():
    print(f"Gathering Golden Trajectories using {MODEL_ID}...\n")
    env = LegacyforgeEnvironment()
    dataset = []

    for ep in range(1, NUM_EPISODES + 1):
        print(f"[{ep}/{NUM_EPISODES}] Starting Episode...")
        trajectory = run_episode(env)
        
        if trajectory:
            dataset.append({"conversations": trajectory})
            print(f"--> Saved! Total perfect runs collected: {len(dataset)}\n")
            
            # Save incrementally just in case you stop it early
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
        else:
            print(f"--> Discarded.\n")

if __name__ == "__main__":
    main()