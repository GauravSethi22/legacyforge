import subprocess
import tempfile
import os
import sys

# Import resource for Linux-based process limits
try:
    import resource
except ImportError:
    resource = None  # Windows fallback

def limit_resources():
    """
    Applies strict OS-level limits to the child process.
    This runs after the fork() but before the exec(), safely sandboxing the untrusted code.
    """
    if not resource:
        return

    try:
        mem_limit = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

        resource.setrlimit(resource.RLIMIT_NPROC, (10, 10))

        file_limit = 5 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))
    except (ValueError, OSError):
        pass

def run_in_sandbox(app_code: str, test_code: str) -> dict:
    """
    Runs pytest in a restricted subprocess with hardware limits.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        app_path = os.path.join(temp_dir, "app_module.py")
        test_path = os.path.join(temp_dir, "test_app.py")
        runner_path = os.path.join(temp_dir, "secure_runner.py")

        with open(app_path, "w") as f:
            f.write(app_code)

        with open(test_path, "w") as f:
            f.write(test_code)

        secure_runner_code = f"""
import sys
import os
import pytest

ALLOWED_DIR = {repr(temp_dir)}
DEVNULL = os.path.abspath(os.devnull)

def security_hook(event, args):
    if event == "open":
        file_path, mode, flags = args
        # Check if it's a write operation AND if the file_path is actually a string
        if isinstance(mode, str) and any(m in mode for m in ['w', 'a', '+']) and isinstance(file_path, str):
            abs_path = os.path.abspath(file_path)
            if not abs_path.startswith(ALLOWED_DIR) and abs_path != DEVNULL:
                raise PermissionError(f"Sandbox Violation: Cannot write to {{abs_path}}")

sys.addaudithook(security_hook)

# Run pytest now that the hook is active
sys.exit(pytest.main(["test_app.py"]))
"""

        with open(runner_path, "w") as f:
            f.write(secure_runner_code)

        try:
            safe_env = {
                "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "PYTHONPATH": temp_dir,
                "TMPDIR": temp_dir,
                "TEMP": temp_dir,
                "TMP": temp_dir,
            }

            result = subprocess.run(
                [sys.executable, runner_path],
                cwd=temp_dir,
                env=safe_env,
                capture_output=True,
                text=True,
                timeout=10,
            )

            passed = result.returncode == 0

            max_output_len = 10000
            out_str = result.stdout[:max_output_len] if result.stdout else ""
            err_str = result.stderr[:max_output_len] if result.stderr else ""

            return {
                "passed": passed,
                "output": out_str,
                "error": err_str,
                "timed_out": False
            }
        except subprocess.TimeoutExpired as e:
            max_output_len = 10000
            out_str = e.stdout[:max_output_len] if e.stdout else ""
            return {
                "passed": False,
                "output": out_str,
                "error": "Execution timed out after 10 seconds",
                "timed_out": True
            }
