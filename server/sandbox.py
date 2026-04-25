import subprocess
import tempfile
import os
import sys

def run_in_sandbox(app_code: str, test_code: str) -> dict:
    """
    Runs pytest in a subprocess safely.
    app_code: The FastAPI app code to test.
    test_code: The pytest code to run against app_code.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        app_path = os.path.join(temp_dir, "app_module.py")
        test_path = os.path.join(temp_dir, "test_app.py")
        
        with open(app_path, "w") as f:
            f.write(app_code)
            
        with open(test_path, "w") as f:
            f.write(test_code)
            
        try:
            # Add temp_dir to PYTHONPATH so app_module can be imported
            env = os.environ.copy()
            env["PYTHONPATH"] = temp_dir + os.pathsep + env.get("PYTHONPATH", "")
            
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path],
                cwd=temp_dir,
                env=env,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            passed = result.returncode == 0
            return {
                "passed": passed,
                "output": result.stdout,
                "error": result.stderr,
                "timed_out": False
            }
        except subprocess.TimeoutExpired as e:
            return {
                "passed": False,
                "output": e.stdout.decode() if e.stdout else "",
                "error": "Execution timed out after 10 seconds",
                "timed_out": True
            }
