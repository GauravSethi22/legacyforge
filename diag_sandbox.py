"""
Quick diagnostic: run the sandbox directly and print raw output.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from server.sandbox import run_in_sandbox

with open("server/levels/level1_answer.py") as f:
    agent_code = f.read()

with open("server/levels/level1_tests.py") as f:
    test_code = f.read()

result = run_in_sandbox(agent_code, test_code)
print("passed:", result["passed"])
print("timed_out:", result["timed_out"])
print("OUTPUT:\n", result["output"])
print("ERROR:\n", result["error"])
