import os
from .sandbox import run_in_sandbox
from .challenger import run_challenger

def validate_test(test_code: str, agent_code: str) -> dict:

    # --- 1. DEFINE THE REQUIRED IMPORTS AND SETUP ---
    test_boilerplate = """
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)
"""
    # Combine the boilerplate with the agent's submitted test
    full_test_code = test_boilerplate + "\n" + test_code

    golden_path = os.path.join(os.path.dirname(__file__), "levels", "golden_solution_l1.py")
    try:
        with open(golden_path, "r") as f:
            golden_code = f.read()
    except FileNotFoundError:
        return {"accepted": False, "reward": 0, "reason": "oracle_unavailable",
                "details": f"Golden solution not found at {golden_path}. Cannot validate."}

    # --- 2. USE full_test_code FOR ALL GATES INSTEAD OF test_code ---
    oracle_gate = run_in_sandbox(golden_code, full_test_code)
    if not oracle_gate["passed"]:
        # If it fails against the perfect code, the test logic itself is broken
        return {"accepted": False, "reward": -2, "reason": "broken_logic", "details": oracle_gate["error"] or oracle_gate["output"]}

    # NEW GATE 2: Solvability Check (Does the agent's code pass its own valid test?)
    solvability_gate = run_in_sandbox(agent_code, full_test_code)
    if not solvability_gate["passed"]:
        # The test is good (it passed the oracle), but the agent's code isn't ready yet!
        return {"accepted": False, "reward": -1, "reason": "agent_code_failing", "details": solvability_gate["error"] or solvability_gate["output"]}

    # GATE 3: Challenge Check
    # Run the test against the challenger's buggy code - must FAIL
    gate3 = run_challenger(full_test_code)
    if gate3["passed"]:
        return {"accepted": False, "reward": 0, "reason": "too_easy", "details": "Test did not catch the bugs injected by the challenger."}

    # All 3 gates passed!
    return {"accepted": True, "reward": 3, "reason": "adversarial_success"}
