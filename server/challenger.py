from .sandbox import run_in_sandbox

BUGGY_CHALLENGER_CODE = """
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Bug 1: sync def instead of async def
# Bug 2: skips Pydantic response_model
@app.get("/items/{item_id}")
def read_item(item_id: int):
    if item_id <= 0:
        # Bug 3: uses 400 instead of 422 for non-positive item_id
        raise HTTPException(status_code=400, detail="Item not found")
    # Bug 4: no upper-bound check — returns 200 for any item_id > 0, even id=9999
    # Bug 5: hardcoded name instead of item-specific f"Item {item_id}"
    return {"item_id": item_id, "name": "Test Item"}
"""

def run_challenger(code: str) -> dict:
    """
    Simulates a junior developer submitting buggy code against the test suite.
    """
    result = run_in_sandbox(BUGGY_CHALLENGER_CODE, code)

    # FIX: Check for sandbox crashes and timeouts first!
    if result.get("timed_out", False):
        return {"passed": False, "failed_tests": [], "reason_code": "broken_logic"}

    error_output = result.get("error", "") + result.get("output", "")
    if "SyntaxError" in error_output or "ImportError" in error_output or "NameError" in error_output:
        return {"passed": False, "failed_tests": [], "reason_code": "broken_logic"}

    if result["passed"]:
        # The test suite passed successfully even though the code has bugs!
        # This means the test didn't catch the bugs (it's too easy/permissive).
        return {"passed": True, "failed_tests": [], "reason_code": "too_easy"}
    else:
        # The test suite failed normally, which means it successfully caught the bugs!
        return {"passed": False, "failed_tests": ["caught_bug"], "reason_code": ""}
