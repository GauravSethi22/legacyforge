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
import random  # Added for dynamic levels and diverse hints

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

GROQ_TOKEN = os.environ.get("GROQ_API_KEY", "").strip()

from server.legacyforge_env import LegacyforgeEnvironment
from models import LegacyforgeAction
from server.custom_levels import my_custom_levels # Pulling in the dynamic dataset

MODEL_ID       = "llama-3.3-70b-versatile"
NUM_EPISODES   = 150 # Set this to whatever you need
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
- A module with `app = FastAPI()` that can be imported.
- You MUST use `from fastapi import FastAPI, HTTPException, Header, Query` (and any other necessary imports).
- You MUST use async def.
- You MUST use Pydantic models for request payloads and responses.
- YOU MUST MATCH the exact routes, methods, HTTP status codes, and JSON response shapes of the provided legacy Flask code.

EXAMPLE TARGET STRUCTURE (You MUST adapt this to match the provided legacy code!):
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ExampleResponse(BaseModel):
    message: str

@app.get("/example", response_model=ExampleResponse)
async def read_example():
    return ExampleResponse(message="Success")

ACTION ORDER:
1. Call read_docs
2. Call edit_function
3. Call run_tests
4. If passage < 70%, call code_review to analyze the test failures, THEN call edit_function again.
5. If passage >= 70%, call submit_test

CRITICAL: For edit_function, new_code must be the ENTIRE module -
all imports, app = FastAPI(), models, and route handlers.
NOT just a single function.

STRICT JSON RULES:
1. You must respond with ONLY valid JSON. No conversational text whatsoever.
2. Any Python code inside "new_code" or "test_code" MUST have all newlines properly escaped as \\n.
3. Do NOT use raw/literal newlines inside JSON strings! It will break the parser.

RESPONSE FORMAT (Strict JSON only, no markdown):
{"action": "edit_function", "params": {"name": "module", "new_code": "<INSERT_ENTIRE_MODULE_CODE_HERE>"}}
OR
{"action": "code_review", "params": {"justification": "<EXPLAIN_WHY_TESTS_FAILED>"}}
OR
{"action": "submit_test", "params": {"test_code": "def test_adversarial():\\n    assert client.get('/items/1001').status_code == 404"}}
"""

def extract_json(text: str) -> dict:
    # 1. Clean up markdown fences and conversational text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    clean_text = match.group(0) if match else text.strip()

    # 2. Try the standard JSON parser first
    try:
        return json.loads(clean_text, strict=False)
    except json.JSONDecodeError:
        pass

    # 3. DIRTY FALLBACK: Manually slice out the code string.
    try:
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', clean_text)
        if not action_match:
            raise ValueError("No action found")

        action = action_match.group(1)

        if action == "edit_function":
            code_start_idx = clean_text.find('"new_code"')
            quote_start = clean_text.find('"', code_start_idx + 10)
            quote_end = clean_text.rfind('"', quote_start + 1, clean_text.rfind('}'))

            code = clean_text[quote_start+1 : quote_end]
            code = code.replace("\\n", "\n").replace('\\"', '"')

            return {"action": "edit_function", "params": {"name": "module", "new_code": code}}

        elif action == "submit_test":
            code_start_idx = clean_text.find('"test_code"')
            quote_start = clean_text.find('"', code_start_idx + 11)
            quote_end = clean_text.rfind('"', quote_start + 1, clean_text.rfind('}'))

            code = clean_text[quote_start+1 : quote_end]
            code = code.replace("\\n", "\n").replace('\\"', '"')

            return {"action": "submit_test", "params": {"test_code": code}}

        elif action == "code_review":
            return {"action": "code_review", "params": {"justification": "Tests failed, proceeding to edit."}}
        elif action == "read_docs":
            topic_match = re.search(r'"topic"\s*:\s*"([^"]+)"', clean_text)
            topic = topic_match.group(1) if topic_match else "routing"
            return {"action": "read_docs", "params": {"topic": topic}}
        elif action == "run_tests":
            return {"action": "run_tests", "params": {}}

    except Exception as e:
        raise ValueError(f"Fallback extraction failed: {str(e)}")

    raise ValueError("No valid JSON object containing an action was found.")

def call_model(user_msg: str) -> dict | None:
    payload = {
        "model": MODEL_ID,
        "messages": [{"role": "user", "content": user_msg}],
        "max_tokens": 1024,
        "temperature": 0.7, # Keep high for diverse trajectory generation
    }

    # Much more aggressive backoff for unattended data generation
    backoff_delays = [5, 15, 30, 60, 60]

    for attempt in range(len(backoff_delays) + 1):
        delay = backoff_delays[attempt] if attempt < len(backoff_delays) else None
        try:
            resp = httpx.post(_GROQ_URL, headers=_GROQ_HEADERS, json=payload, timeout=60)
            if resp.status_code == 429:
                sleep_time = delay or 60
                print(f"    [!] Rate limited. Waiting {sleep_time}s...")
                time.sleep(sleep_time)
                continue

            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return extract_json(raw)

        except Exception as e:
            print(f"    [!] API/Network Error (Attempt {attempt+1}): {e}")
            if delay is not None:
                print(f"    [*] Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("    [!] Exhausted all retries.")
                return None

    return None

def model_response_to_action(response: dict) -> LegacyforgeAction | None:
    try:
        action_type = response.get("action", "")
        params      = response.get("params", {}) or {}

        valid_types = {"read_docs", "edit_function", "run_tests",
                       "code_review", "submit_test"}
        if action_type not in valid_types:
            return None

        return LegacyforgeAction(
            action_type=action_type,
            target=params.get("topic") or params.get("name") or params.get("justification"),
            code=params.get("new_code") or params.get("test_code"),
        )
    except Exception:
        return None

def run_dynamic_episode(env: LegacyforgeEnvironment, custom_level: dict = None) -> list | None:
    # Pass the config dictionary to the environment
    obs = env.reset(level_config=custom_level)
    action_history = []
    trajectory = []
    last_result = "None"

    for step_idx in range(1, MAX_ACTIONS + 1):
        time.sleep(3) # Groq rate limit buffer
        current_state = env.state
        history_str = " -> ".join(action_history[-5:]) if action_history else "None"

        last_action = action_history[-1] if action_history else None
        read_docs_count = current_state["read_docs_count"]

        # Dynamic Hint Logic integrated with code_review
        if last_action == "read_docs" and read_docs_count >= 1:
            force_hint = "You MUST call edit_function now. Do NOT call read_docs again."
        elif last_action == "edit_function":
            force_hint = "You MUST call run_tests now. Do NOT call anything else."
        elif last_action == "run_tests" and current_state['test_passage_rate'] >= 0.70:
            force_hint = "Passage rate is above 70%. You MUST call submit_test now."
        elif last_action == "run_tests" and current_state['test_passage_rate'] < 0.70:
            force_hint = "Tests failing! Call code_review to analyze the pytest errors. Do NOT call edit_function yet."
        elif last_action == "code_review":
            force_hint = "Now that you analyzed the errors, call edit_function and provide the ENTIRE FIXED FASTAPI MODULE."
        elif last_action == "submit_test":
            force_hint = "Adversarial test rejected. You MUST call submit_test again and provide valid Pytest code in the 'test_code' parameter."
        else:
            # Randomized starting hints for diverse SFT data
            start_choices = [
                "Start by calling read_docs with topic 'routing'.",
                "Start by calling read_docs with topic 'pydantic'.",
                "Start by calling read_docs with topic 'async'.",
                "Skip the docs for now and start directly by calling edit_function. Guess the FastAPI structure.",
                "Start by calling read_docs with topic 'exceptions' to see how to handle the 404 and 422 errors."
            ]
            force_hint = random.choice(start_choices)

        user_msg = SYSTEM_PROMPT_TEMPLATE + f"""
Recent actions: {history_str}
read_docs calls this episode: {read_docs_count} / 2 max

HINT: {force_hint}

Current legacy code:
{obs.legacy_code[:800]}

Current passage rate: {current_state['test_passage_rate']:.2f}
Current phase: {current_state['phase']}
Last action result: {last_result}

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

        # -------------------------------------------------------------
        # Update last_result based on the environment's response to the action
        # -------------------------------------------------------------
        if action.action_type == "submit_test":
            val = obs.info.get("validation", {})
            if not val.get("accepted"):
                # Grab the last 500 characters of the Pytest log to show the model the error
                error_log = str(val.get('details', ''))[-500:]
                last_result = f"Adversarial test rejected! Pytest output:\n{error_log}"
        elif action.action_type == "run_tests":
            last_result = f"Tests executed. Passage: {env.state['test_passage_rate']:.2f}"
        else:
            last_result = f"Action {action.action_type} executed."

        print(f"  Step {step_idx:2d} | Action: {action.action_type:<15} | Passage: {env.state['test_passage_rate']:.2f} | Phase: {env.state['phase']}")

        if obs.done:
            # Check if the episode ended because the adversarial test was accepted
            if action.action_type == "submit_test" and obs.info.get("validation", {}).get("accepted"):
                print(f"  ✅ Perfect run! Tests passed and adversarial test succeeded.")
                return trajectory
            else:
                print(f"  ❌ Episode ended. (Timeout, crash, or failed adversarial test)")
                return None

    print(f"  ❌ Timed out after {MAX_ACTIONS} steps.")
    return None

def main():
    print(f"Gathering Golden Trajectories using {MODEL_ID}...\n")
    env = LegacyforgeEnvironment()
    dataset = []

    for ep in range(1, NUM_EPISODES + 1):
        # Pick a random custom level for this episode
        level_to_play = random.choice(my_custom_levels)
        level_name = level_to_play["level_name"] if level_to_play and "level_name" in level_to_play else "Level 1"

        print(f"[{ep}/{NUM_EPISODES}] Starting Episode... (Task: {level_name})")

        # Safe fallback to level 1
        if level_to_play and level_to_play.get("flask_code") is None:
            level_to_play = None

        trajectory = run_dynamic_episode(env, custom_level=level_to_play)

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
