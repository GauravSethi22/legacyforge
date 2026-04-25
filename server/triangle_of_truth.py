import os
from .sandbox import run_in_sandbox
from .challenger import run_challenger

def validate_test(test_code: str, agent_code: str) -> dict:
    # Gate 1: Solvability Check
    # Run the test against the agent's own migrated code - must PASS
    gate1 = run_in_sandbox(agent_code, test_code)
    if not gate1["passed"]:
        return {"accepted": False, "reward": -1, "reason": "unfair_test", "details": gate1["error"] or gate1["output"]}
        
    # Gate 2: Oracle Check
    # Run the test against the hidden golden solution - must PASS
    golden_path = os.path.join(os.path.dirname(__file__), "levels", "golden_solution_l1.py")
    try:
        with open(golden_path, "r") as f:
            golden_code = f.read()
    except FileNotFoundError:
        return {"accepted": False, "reward": 0, "reason": "oracle_unavailable",
                "details": f"Golden solution not found at {golden_path}. Cannot validate."}
        
    gate2 = run_in_sandbox(golden_code, test_code)
    if not gate2["passed"]:
        return {"accepted": False, "reward": -2, "reason": "broken_logic", "details": gate2["error"] or gate2["output"]}
        
    # Gate 3: Challenge Check
    # Run the test against the challenger's buggy code - must FAIL
    # run_challenger returns passed=True if the tests passed (failed to catch bugs)
    gate3 = run_challenger(test_code)
    if gate3["passed"]:
        return {"accepted": False, "reward": 0, "reason": "too_easy", "details": "Test did not catch the bugs injected by the challenger."}
        
    # All 3 gates passed!
    return {"accepted": True, "reward": 3, "reason": "adversarial_success"}
