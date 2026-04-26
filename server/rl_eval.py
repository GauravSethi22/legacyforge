"""
rl_eval.py
Runs the locally trained legacyforge-8b-rl-final model via Unsloth
against LegacyforgeEnvironment for evaluation and logs all reward components.
"""
import json
import os
import sys
import time
import re
import torch
from unsloth import FastLanguageModel

# Ensure the environment can find your custom modules
from custom_levels import my_custom_levels

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from server.legacyforge_env import LegacyforgeEnvironment  # type: ignore
from models import LegacyforgeAction  # type: ignore

# --- CONFIGURATION ---
MODEL_PATH     = os.path.join(_ROOT, "legacyforge-8b-rl-final")
NUM_EPISODES   = 4 # Exactly 4 episodes
MAX_ACTIONS    = 20
OUTPUT_FILE    = os.path.join(_HERE, "rl_eval_results.json")
MAX_SEQ_LENGTH = 2048

print(f"\nLoading local RL model from '{MODEL_PATH}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
# CRITICAL: Switch to 2x faster inference mode
FastLanguageModel.for_inference(model)
print("Model loaded successfully!\n")


SYSTEM_PROMPT_TEMPLATE = """\
You are migrating a Flask app to FastAPI. Your goal: make ALL tests pass.

THE TESTS EXPECT:
- A module with `app = FastAPI()` that can be imported.
- You MUST use `from fastapi import FastAPI, HTTPException, Header, Query` (and any other necessary imports).
- You MUST use async def.
- You MUST use Pydantic models for request payloads and responses.
- YOU MUST MATCH the exact routes, methods, HTTP status codes, and JSON response shapes of the provided legacy Flask code.

CRITICAL FOR submit_test:
- A global `client` (FastAPI TestClient) is ALREADY provided in the test environment.
- DO NOT import `TestClient`, `app`, or your module inside the test code. Just write the test function using the global `client`.

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
6. If submit_test fails, read the "Last action result" to see the Pytest error. Fix the bug in your test code and submit again. DO NOT edit the API.

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
    # 1. Try to parse the entire text as JSON first
    try:
        return json.loads(text.strip(), strict=False)
    except json.JSONDecodeError:
        pass

    # 2. If there is conversational text, find the JSON block using regex
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")

    json_str = match.group(0)

    # 3. Let the standard json module handle all escaping
    try:
        parsed = json.loads(json_str, strict=False)
        if "action" not in parsed:
            raise ValueError("Parsed JSON missing 'action' key.")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Extracted block is not valid JSON: {str(e)}")

def call_model(user_msg: str) -> dict | None:
    messages = [{"role": "user", "content": user_msg}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    try:
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=1500, # Increased to match RL config
            use_cache=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

        raw = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        return extract_json(raw)

    except (json.JSONDecodeError, ValueError) as e:
        return {"_raw": raw, "_error": str(e)}
    except Exception as e:
        print(f"\n[DEBUG] Local Inference Error: {e}")
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

def run_episode(episode_num: int, env: LegacyforgeEnvironment, custom_level: dict = None) -> dict:
    print(f"\n{'═'*60}")
    level_name = custom_level["level_name"] if custom_level else "Level 1 (Default)"
    print(f" EPISODE {episode_num} / {NUM_EPISODES} | Task: {level_name}")
    print(f"{'═'*60}")

    obs = env.reset(level_config=custom_level)
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
        "steps":             [],
    }

    last_result = "None"
    action_history = []
    last_submit_reason = None

    for step_idx in range(1, MAX_ACTIONS + 1):
        current_state = env.state
        history_str = " -> ".join(action_history[-5:]) if action_history else "None"

        last_action = action_history[-1] if action_history else None
        read_docs_count = current_state["read_docs_count"]

        # Dynamic hints exactly as used in data generation
        if last_action == "read_docs" and read_docs_count >= 1:
            force_hint = "You MUST call edit_function now. Do NOT call read_docs again."
        elif last_action == "edit_function":
            force_hint = "You MUST call run_tests now. Do NOT call anything else."
        elif last_action == "run_tests" and current_state['test_passage_rate'] >= 0.70:
            force_hint = "Passage rate is above 70%. You MUST call submit_test now."
        elif last_action == "run_tests" and current_state['test_passage_rate'] < 0.70:
            force_hint = "Tests failing! Call code_review to analyze the errors."
        elif last_action == "code_review":
            force_hint = "Now that you analyzed the errors, call edit_function and provide the ENTIRE FIXED FASTAPI MODULE. Do NOT use placeholders."
        elif last_action == "submit_test":
            if last_submit_reason == "agent_code_failing":
                force_hint = "Your FastAPI code has a bug! It failed the valid test you just wrote (error shown below). You MUST call edit_function to fix your API implementation."
            elif last_submit_reason == "broken_logic":
                force_hint = "Your test code has a bug! It failed against the golden solution. DO NOT call edit_function. Fix your test code using the global `client` and call submit_test again."
            elif last_submit_reason == "too_easy":
                force_hint = "Your test is too weak and didn't catch the hidden bugs. DO NOT call edit_function. Write a stricter test with more assertions and call submit_test again."
            else:
                force_hint = "Adversarial test rejected! Fix it and call submit_test again."
        else:
            force_hint = "Start by calling read_docs with topic async or routing."

        user_msg = SYSTEM_PROMPT_TEMPLATE + f"""
Recent actions: {history_str}
read_docs calls this episode: {read_docs_count} / 2 max

HINT: {force_hint}

Current legacy code:
{obs.legacy_code}

Current passage rate: {current_state['test_passage_rate']:.2f}
Current phase: {current_state['phase']}
Last action result: {last_result}

Respond with a single JSON object only.
"""
        response = call_model(user_msg)

        if response is None:
            last_result = "Inference failure"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ INFERENCE FAIL │ ")
            continue

        if "_error" in response or not response:
            metrics["parse_errors"] += 1
            err_msg = response.get('_error', 'Invalid JSON') if response else 'Empty response'
            last_result = f"JSON Parse Error: {err_msg}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ PARSE ERROR    │ {err_msg[:60]}")
            continue

        action = model_response_to_action(response)
        if action is None:
            metrics["parse_errors"] += 1
            last_result = f"Invalid action: {response.get('action', '?')}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ BAD ACTION     │ {last_result[:60]}")
            continue

        action_history.append(action.action_type)

        try:
            obs = env.step(action)
        except Exception as exc:
            last_result = f"Environment Exception: {exc}"
            print(f" Step {step_idx:2d} │ Phase {current_state['phase']} │ Passage {current_state['test_passage_rate']:5.2f} │ ❌ ENV ERROR      │ {str(exc)[:60]}")
            continue

        if action.action_type == "submit_test":
            val = obs.info.get("validation", {})
            last_submit_reason = val.get("reason", "unknown")
            if not val.get("accepted"):
                error_log = str(val.get('details', ''))[-500:]
                last_result = f"Adversarial test rejected (Reason: {last_submit_reason})! Pytest output:\n{error_log}"
            else:
                last_result = "Adversarial test succeeded!"
        elif action.action_type == "run_tests":
            last_result = f"Tests executed. Passage: {env.state['test_passage_rate']:.2f}"
        elif action.action_type == "edit_function":
            last_result = f"API Code updated successfully."
        else:
            last_result = f"Action {action.action_type} executed."

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
        if not metrics["episode_completed"]:
            print(f"Episode timed out after {MAX_ACTIONS} steps")

    print(f"Parse errors: {metrics['parse_errors']}")
    return metrics


def main():
    print(f"\nLegacyForge RL Model Evaluation (4 Levels)")
    print(f"Model : {MODEL_PATH}")
    print(f"Episodes: {NUM_EPISODES}  |  Max actions/episode: {MAX_ACTIONS}")
    print(f"Output: {OUTPUT_FILE}\n")

    env     = LegacyforgeEnvironment()
    results = []

    for ep in range(1, NUM_EPISODES + 1):
        # Deterministically select level ep-1 to cover Level 1 to 4 precisely
        level_to_play = my_custom_levels[ep - 1]

        if level_to_play.get("flask_code") is None:
            level_to_play = None

        metrics = run_episode(ep, env, custom_level=level_to_play)
        results.append(metrics)

    n = len(results)
    if n == 0:
        return

    avg_reward    = sum(r["total_reward"]      for r in results) / n
    avg_passage   = sum(r["test_passage_rate"] for r in results) / n
    avg_strategy  = sum(r["strategy_quality"]  for r in results) / n
    avg_actions   = sum(r["actions_taken"]     for r in results) / n
    phase2_rate   = sum(1 for r in results if r["phase2_unlocked"])
    oracle_hits   = sum(1 for r in results if r["oracle_penalty"] > 0)
    completed     = sum(1 for r in results if r["episode_completed"])
    tot_parse_err = sum(r["parse_errors"] for r in results)

    print(f"\n{'═'*60}")
    print(" RL TRAINED EVALUATION SUMMARY")
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
    print(f"{'═'*60}")

    output = {
        "model":    MODEL_PATH,
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
        },
        "episodes_detail": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
