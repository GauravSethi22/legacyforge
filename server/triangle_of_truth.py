import os
from .sandbox import run_in_sandbox
from .challenger import run_challenger

def validate_test(test_code: str, agent_code: str, golden_code: str) -> dict:

    test_boilerplate = """
import pytest
from fastapi.testclient import TestClient
from app_module import app

client = TestClient(app)
"""
    full_test_code = test_boilerplate + "\n" + test_code

    oracle_gate = run_in_sandbox(golden_code, full_test_code)
    if not oracle_gate["passed"]:
        return {"accepted": False, "reward": -2, "reason": "broken_logic", "details": oracle_gate["error"] or oracle_gate["output"]}

    # 2. Solvability Gate
    solvability_gate = run_in_sandbox(agent_code, full_test_code)
    if not solvability_gate["passed"]:
        return {"accepted": False, "reward": -1, "reason": "agent_code_failing", "details": solvability_gate["error"] or solvability_gate["output"]}

    # 3. Challenge Check
    gate3 = run_challenger(full_test_code)
    if gate3["passed"]:
        return {"accepted": False, "reward": 0, "reason": "too_easy", "details": "Test did not catch the bugs injected by the challenger."}

    return {"accepted": True, "reward": 3, "reason": "adversarial_success"}
