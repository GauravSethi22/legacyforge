"""
baseline_eval.py
Runs llama-3.1-8b-instant via Groq API
against LegacyforgeEnvironment for 10 episodes and logs all reward components.
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
if not GROQ_TOKEN:
    print(
        "\nERROR: GROQ_API_KEY environment variable is not set.\n"
        "Get a FREE key (no credit card) at: https://console.groq.com\n"
        "Then run:\n"
        "  PowerShell : $env:GROQ_API_KEY = 'gsk_...'; uv run python server/baseline_eval.py\n"
        "  CMD        : set GROQ_API_KEY=gsk_... && uv run python server/baseline_eval.py\n"
    )
    sys.exit(1)

from server.legacyforge_env import LegacyforgeEnvironment  # type: ignore
from models import LegacyforgeAction  # type: ignore

MODEL_ID       = "llama-3.1-8b-instant"
NUM_EPISODES   = 3
MAX_ACTIONS    = 20
OUTPUT_FILE    = os.path.join(_HERE, "baseline_results.json")

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
1. Call read_docs
2. Call edit_function
3. Call run_tests
4. If passage < 70%, call code_review to analyze the test failures, THEN call edit_function again.
5. If passage >= 70%, call submit_test

CRITICAL: For edit_function, new_code must be the ENTIRE module -
all imports, app = FastAPI(), models, and route handlers.
NOT just a single function.

RESPONSE FORMAT (Strict JSON only, no markdown):
{"action": "edit_function", "params": {"name": "module", "new_code": "<INSERT_ENTIRE_MODULE_CODE_HERE_AS_ESCAPED_STRING>"}}
"""

def extract_json(text: str) -> dict:
    # Strip markdown fences
    text = re.sub(r"```json|```", "", text).strip()

    # Find all potential JSON objects using regex
    # This looks for matching curly braces (handling basic nesting)
    #json_pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    # Try the simple approach first: if the whole text parses, use it
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If not, extract all { ... } blocks
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

    # Search through the extracted blocks for a valid action
    last_valid_json = None
    for block in blocks:
        try:
            parsed = json.loads(block)
            last_valid_json = parsed
            # If we find a block that specifically has the 'action' key, return it immediately
            if isinstance(parsed, dict) and "action" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    if last_valid_json is not None:
        return last_valid_json

    raise ValueError("No valid JSON object containing an action was found.")

def call_model(user_msg: str, rate_limit_hits: list) -> dict | None:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 1024,
        "temperature": 0.2,
    }

    backoff_delays = [5, 15, 30]

    for attempt in range(len(backoff_delays) + 1):
        delay = backoff_delays[attempt] if attempt < len(backoff_delays) else None
        raw = ""   # BUG 4 FIX: ensure raw is always defined before the except clause
        try:
            resp = httpx.post(
                _GROQ_URL,
                headers=_GROQ_HEADERS,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                if delay is not None:
                    rate_limit_hits[0] += 1
                    time.sleep(delay)
                    continue
                else:
                    return None  # all retries exhausted
            resp.raise_for_status()

            raw = resp.json()["choices"][0]["message"]["content"].strip()
            return extract_json(raw)

        except (json.JSONDecodeError, ValueError) as e:
            return {"_raw": raw, "_error": str(e)}
        except httpx.HTTPStatusError as e:
            print(f"\n[DEBUG] HTTP Error: {e.response.status_code} - {e.response.text}")
            if delay is not None:
                time.sleep(delay)
                continue
            return None
        except Exception as e:
            print(f"\n[DEBUG] Request Error: {e}")
            if delay is not None:
                time.sleep(delay)
                continue
            return None

    return None  # exhausted all retries

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

def run_episode(episode_num: int, env: LegacyforgeEnvironment) -> dict:
    print(f"\n{'═'*60}")
    print(f" EPISODE {episode_num} / {NUM_EPISODES}")
    print(f"{'═'*60}")

    obs = env.reset(level=1)
    metrics = {
        "episode":           episode_num,
        "total_reward":      0.0,
        "migration_success": 0.0,
        "strategy_quality":  0.0,
        "adversarial_bonus": 0.0,
        "oracle_penalty":    0.0,
        "test_passage_rate": 0.0,
        "phase_reached":     1,
        "actions_taken":     0,
        "phase2_unlocked":   False,
        "episode_completed": False,
        "parse_errors":      0,
        "rate_limit_hits":   0,
        "steps":             [],
    }

    last_result = "None"
    action_history = []
    rl_hits = [0]
    consecutive_failures = 0

    for step_idx in range(1, MAX_ACTIONS + 1):
        time.sleep(3) # Mandatory sleep between EVERY step

        current_state = env.state
        history_str = " -> ".join(action_history[-5:]) if action_history else "None"

        last_action = action_history[-1] if action_history else None
        # BUG 5 FIX: read from env.state — single source of truth
        read_docs_count = current_state["read_docs_count"]

        if last_action == "read_docs" and read_docs_count >= 1:
            force_hint = "You MUST call edit_function now. Do NOT call read_docs again."
        elif last_action == "edit_function":
            force_hint = "You MUST call run_tests now. Do NOT call anything else."
        elif last_action == "run_tests" and current_state['test_passage_rate'] >= 0.70:
            force_hint = "Passage rate is above 70%. You MUST call submit_test now."
        elif last_action == "run_tests" and current_state['test_passage_rate'] < 0.70:
            # ---> THIS IS THE LINE TO REPLACE <---
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
Last action result: {last_result}

Respond with a single JSON object only.
"""

        response = call_model(user_msg, rl_hits)

        if response is None:
            consecutive_failures += 1
            last_result = "API failure"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ API FAILURE    │ ({consecutive_failures} consecutive)")
            if consecutive_failures >= 3:
                print(f"Episode died ({consecutive_failures} consecutive API failures)")
                break
            continue

        consecutive_failures = 0

        if "_error" in response or not response:
            metrics["parse_errors"] += 1
            err_msg = response.get('_error', 'Invalid JSON') if response else 'Empty response'
            raw_preview = str(response.get('_raw', ''))[:80] if response else ''
            last_result = f"JSON Parse Error: {err_msg}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ PARSE ERROR    │ {err_msg[:60]}")
            continue

        action = model_response_to_action(response)
        if action is None:
            metrics["parse_errors"] += 1
            last_result = f"Invalid action: {response.get('action', '?')}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ BAD ACTION     │ {last_result[:60]}")
            continue

        if action.action_type == "edit_function":
            print(f"\n[DEBUG] FULL Code Injected:\n{'-'*40}\n{action.code}\n{'-'*40}\n")

        action_history.append(action.action_type)


        try:
            obs = env.step(action)
        except Exception as exc:
            last_result = f"Environment Exception: {exc}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ ENV ERROR      │ {str(exc)[:60]}")
            continue

        reward = obs.reward
        rb     = obs.reward_breakdown or {}

        metrics["total_reward"]      += reward
        metrics["migration_success"] += rb.get("migration_success", 0.0)
        metrics["adversarial_bonus"] += rb.get("adversarial_bonus", 0.0)
        metrics["oracle_penalty"]    += rb.get("oracle_penalty", 0.0)
        if rb.get("strategy_quality", 0.0) > 0:
            metrics["strategy_quality"] = rb["strategy_quality"]

        metrics["actions_taken"] = step_idx

        print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ {action.action_type:<15} │ reward {reward:>+6.2f} │ total {metrics['total_reward']:>+6.2f}")

        last_result = f"Reward: {reward:.2f}, Info: {obs.info}"

        metrics["steps"].append({
            "step":        step_idx,
            "action":      action.action_type,
            "reward":      round(reward, 4),
            "breakdown":   {k: round(v, 4) for k, v in rb.items()},
        })

        state_now = env.state
        metrics["test_passage_rate"] = state_now["test_passage_rate"]
        metrics["phase_reached"]     = state_now["phase"]
        if state_now["phase"] == 2:
            metrics["phase2_unlocked"] = True

        if obs.done:
            metrics["episode_completed"] = True
            print(f"Episode complete in {step_idx} steps")
            break

    else:
        if not metrics["episode_completed"] and consecutive_failures < 3:
            print(f"Episode timed out after {MAX_ACTIONS} steps")

    metrics["rate_limit_hits"] = rl_hits[0]

    if rl_hits[0] > 0:
        print(f"Rate limited {rl_hits[0]}x this episode")
    print(f"Parse errors: {metrics['parse_errors']}")

    return metrics


def main():
    print(f"\nLegacyForge Baseline Evaluation")
    print(f"Model : {MODEL_ID}")
    print(f"Episodes: {NUM_EPISODES}  |  Max actions/episode: {MAX_ACTIONS}")
    print(f"Output: {OUTPUT_FILE}\n")

    env     = LegacyforgeEnvironment()
    results = []

    for ep in range(1, NUM_EPISODES + 1):
        metrics = run_episode(ep, env)
        results.append(metrics)

    n = len(results)
    avg_reward    = sum(r["total_reward"]      for r in results) / n
    avg_passage   = sum(r["test_passage_rate"] for r in results) / n
    avg_strategy  = sum(r["strategy_quality"]  for r in results) / n
    avg_actions   = sum(r["actions_taken"]     for r in results) / n
    phase2_rate   = sum(1 for r in results if r["phase2_unlocked"])
    oracle_hits   = sum(1 for r in results if r["oracle_penalty"] > 0)
    completed     = sum(1 for r in results if r["episode_completed"])
    tot_parse_err = sum(r["parse_errors"] for r in results)
    tot_rl_hits   = sum(r["rate_limit_hits"] for r in results)

    print(f"\n{'═'*60}")
    print(" BASELINE EVALUATION SUMMARY")
    print(f"{'═'*60}")
    print(f" Episodes run          : {n}")
    print(f" Avg total reward      : {avg_reward:+.3f}")
    print(f" Avg test passage rate : {avg_passage:.3f}")
    print(f" Avg strategy quality  : {avg_strategy:.3f}")
    print(f" Phase 2 unlock rate   : {phase2_rate} / {n}")
    print(f" Episodes completed    : {completed} / {n}")
    print(f" Oracle penalty hits   : {oracle_hits} / {n}")
    print(f" Avg actions/episode   : {avg_actions:.1f}")
    print(f" Total parse errors    : {tot_parse_err}")
    print(f" Total rate limit hits : {tot_rl_hits}")
    print(f"{'═'*60}")
    print(f"\n OFFICIAL BASELINE SCORES (save these before training)")
    print(f"────────────────────────────────────────────────────────────")
    print(f" avg_total_reward      : {avg_reward:.3f}")
    print(f" avg_test_passage_rate : {avg_passage:.3f}")
    print(f" avg_strategy_quality  : {avg_strategy:.3f}")
    print(f" phase2_unlock_rate    : {(phase2_rate/n)*100:.2f}%")
    print(f"{'═'*60}\n")

    output = {
        "model":    MODEL_ID,
        "episodes": NUM_EPISODES,
        "summary": {
            "avg_total_reward":    round(avg_reward,   4),
            "avg_test_passage":    round(avg_passage,  4),
            "avg_strategy_quality": round(avg_strategy, 4),
            "avg_actions":         round(avg_actions,  2),
            "phase2_unlock_rate":  f"{phase2_rate} / {n}",
            "episodes_completed":  f"{completed} / {n}",
            "oracle_penalty_rate": f"{oracle_hits} / {n}",
            "total_parse_errors":  tot_parse_err,
            "total_rate_limit_hits": tot_rl_hits,
        },
        "episodes_detail": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
