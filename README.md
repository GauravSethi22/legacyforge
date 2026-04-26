---
title: Legacyforge Environment Server
emoji: 🏏
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# LegacyForge

> **A reinforcement learning environment for training AI agents to migrate legacy Flask applications to modern FastAPI.**

LegacyForge is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible environment that presents an AI agent with real-world Flask codebases and challenges it to rewrite them as idiomatic, production-grade FastAPI applications — guided by documentation, test feedback, and reward signals.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Reward System](#reward-system)
- [Levels](#levels)
- [Running the Server](#running-the-server)
- [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)

---

## Overview

LegacyForge frames code migration as a sequential decision-making problem. At each step, an agent observes a legacy Flask codebase and chooses one of five actions: read documentation, edit a function, run tests, request a code review, or submit a solution. The environment evaluates correctness, style, and adherence to FastAPI idioms, then returns a structured reward.

This makes LegacyForge suitable for:

- **Supervised Fine-Tuning (SFT)** — golden trajectories are included for imitation learning
- **Reinforcement Learning from test signals** — agents learn to pass pytest suites without human labels
- **Evaluation** — baseline and RL eval scripts measure migration quality across levels

---

## How It Works

```
Agent observes legacy Flask code
        │
        ▼
 Chooses an action:
  ├─ read_docs      → Retrieves relevant FastAPI documentation snippet
  ├─ edit_function  → Rewrites a function in the codebase
  ├─ run_tests      → Executes the pytest suite; returns stdout + pass/fail
  ├─ code_review    → Gets structured feedback on the current migration
  └─ submit_test    → Final submission; triggers full evaluation + reward
        │
        ▼
 Receives observation: updated code, docs, migration history, reward
        │
        ▼
 Repeat until done (submit_test or max steps)
```

The environment tracks a `HistoryAggregator` that categorises recurring error types (async, pydantic, routing, lifespan, middleware) to shape reward signals across episodes.

---

## Quick Start

### 1. Install

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### 2. Build the Docker image

```bash
docker build -t legacyforge-env:latest -f Dockerfile .
```

### 3. Run an episode

```python
from legacyforge import LegacyforgeAction, LegacyforgeEnv

client = LegacyforgeEnv.from_docker_image("legacyforge-env:latest")

try:
    obs = client.reset()
    print("Legacy code:\n", obs.observation.legacy_code)

    # Read the routing docs
    result = client.step(LegacyforgeAction(action_type="read_docs", target="routing"))
    print("Docs:\n", result.observation.docs)

    # Rewrite the endpoint
    fastapi_code = """
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id <= 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "name": "Test Item"}
"""
    result = client.step(LegacyforgeAction(action_type="edit_function", code=fastapi_code))

    # Run tests
    result = client.step(LegacyforgeAction(action_type="run_tests"))
    print("Test output:\n", result.observation.info.get("test_output", ""))

    # Submit
    result = client.step(LegacyforgeAction(action_type="submit_test"))
    print("Final reward:", result.reward)
    print("Reward breakdown:", result.observation.reward_breakdown)

finally:
    client.close()
```

---

## Action Space

Actions are defined in `models.py` as a `LegacyforgeAction` Pydantic model.

| `action_type`   | `target` (optional)         | `code` (optional)     | Description                                          |
|-----------------|-----------------------------|-----------------------|------------------------------------------------------|
| `read_docs`     | doc key (e.g. `"routing"`)  | —                     | Returns a FastAPI documentation snippet              |
| `edit_function` | function name               | New FastAPI code      | Replaces or updates a function in the codebase       |
| `run_tests`     | —                           | —                     | Runs pytest and returns stdout, exit code            |
| `code_review`   | —                           | —                     | Returns structured feedback on the current migration |
| `submit_test`   | —                           | —                     | Final evaluation — ends the episode                  |

**Available documentation keys** for `read_docs`:

| Key                | Description                                         |
|--------------------|-----------------------------------------------------|
| `routing`          | Path parameters and endpoint definitions            |
| `pydantic`         | Request/response models with `BaseModel`            |
| `async`            | `async def` usage in endpoints                      |
| `background_tasks` | `BackgroundTasks` for deferred work                 |
| `lifespan`         | Startup/shutdown with `@asynccontextmanager`        |
| `exceptions`       | `HTTPException` usage and status codes              |

---

## Observation Space

Each step returns a `LegacyforgeObservation`:

| Field                       | Type            | Description                                           |
|-----------------------------|-----------------|-------------------------------------------------------|
| `legacy_code`               | `str`           | Current state of the codebase being migrated          |
| `docs`                      | `str`           | Documentation returned by the last `read_docs` action |
| `migration_history_summary` | `str`           | Summary of actions taken so far in the episode        |
| `level`                     | `int \| str`    | Current environment level                             |
| `reward`                    | `float \| None` | Reward from the last action                           |
| `reward_breakdown`          | `dict \| None`  | Per-criterion breakdown of the reward                 |
| `info`                      | `dict \| None`  | Extra info (e.g. `test_output`, `error_messages`)     |
| `done`                      | `bool`          | `True` when the episode has ended                     |

---

## Reward System

The reward is computed at `submit_test` and broken down across multiple criteria:

- **Test pass rate** — fraction of pytest assertions passing
- **Async correctness** — all endpoints use `async def`
- **Pydantic usage** — request/response models declared with `BaseModel`
- **Route correctness** — decorator syntax matches FastAPI conventions
- **Lifespan compliance** — no deprecated `@app.on_event` usage
- **Style penalties** — deductions for leftover Flask imports or patterns

Intermediate steps (e.g. `run_tests`) may return partial rewards to guide exploration.

---

## Levels

LegacyForge ships with multiple levels of increasing complexity, defined in `server/custom_levels.py`:

| Level | Name                   | Key Concepts                                      |
|-------|------------------------|---------------------------------------------------|
| 1     | Basic Routing          | `GET` endpoints, path parameters, `HTTPException` |
| 2     | CRUD and POST Payloads | `POST`, `GET`, Pydantic request/response models   |
| 3+    | Extensible             | Add your own in `custom_levels.py`                |

Each level provides:
- A **Flask source file** for the agent to migrate
- A **pytest test suite** that defines the migration target
- A **golden solution** used for SFT data generation and evaluation

### Adding a Custom Level

```python
# In server/custom_levels.py
my_custom_levels.append({
    "level_name": "Level 3: Auth Middleware",
    "test_suite_size": 5,
    "flask_code": "...",   # Starting Flask code
    "test_code": "...",    # pytest suite
    "golden_code": "...",  # Reference FastAPI solution
})
```

---

## Running the Server

### Locally (development)

```bash
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Via Docker

```bash
docker build -t legacyforge-env:latest -f Dockerfile .
docker run -p 8000:8000 legacyforge-env:latest
```

### Server endpoints

| Endpoint  | Method    | Description                                |
|-----------|-----------|--------------------------------------------|
| `/reset`  | `POST`    | Start a new episode                        |
| `/step`   | `POST`    | Execute an action                          |
| `/state`  | `GET`     | Current environment state                  |
| `/schema` | `GET`     | Action and observation JSON schemas        |
| `/ws`     | WebSocket | Persistent low-latency session             |
| `/web`    | `GET`     | Interactive web UI                         |
| `/docs`   | `GET`     | OpenAPI / Swagger interface                |
| `/health` | `GET`     | Health check                               |

---

## Deploying to Hugging Face Spaces

```bash
# From the project root (where openenv.yaml lives)
openenv push

# With options
openenv push --repo-id my-org/legacyforge --private
```

**Available flags:**

| Flag              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `--directory, -d` | Directory containing `openenv.yaml` (default: current dir)   |
| `--repo-id, -r`   | Target repo in `username/repo-name` format                   |
| `--base-image, -b`| Override the Dockerfile `FROM` image                         |
| `--private`       | Deploy as a private Space (default: public)                  |

After deployment, the Space is available at `https://huggingface.co/spaces/<repo-id>` with the web UI at `/web`.

---

## Advanced Usage

### Connect to an existing server

```python
from legacyforge import LegacyforgeEnv

env = LegacyforgeEnv(base_url="http://localhost:8000")
result = env.reset()
```

> Note: `env.close()` will **not** stop the server when connecting to an existing instance.

### Context manager

```python
with LegacyforgeEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    result = env.step(LegacyforgeAction(action_type="read_docs", target="async"))
```

### Concurrent sessions

Enable multiple concurrent WebSocket sessions by increasing `max_concurrent_envs` in `server/app.py`:

```python
app = create_app(
    LegacyforgeEnvironment,
    LegacyforgeAction,
    LegacyforgeObservation,
    max_concurrent_envs=4,
)
```

Then run multiple agents in parallel:

```python
from concurrent.futures import ThreadPoolExecutor
from legacyforge import LegacyforgeAction, LegacyforgeEnv

def run_episode(client_id: int):
    with LegacyforgeEnv(base_url="http://localhost:8000") as env:
        env.reset()
        for _ in range(5):
            env.step(LegacyforgeAction(action_type="run_tests"))
        result = env.step(LegacyforgeAction(action_type="submit_test"))
        return client_id, result.reward

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

### Running evals

```bash
# Baseline eval (rule-based agent)
python server/baseline_eval.py

# RL eval
python server/rl_eval.py

# Generate SFT training data from golden solutions
python server/generate_training_data.py

# Generate golden trajectories
python server/generate_golden_data.py
```

## Blog
Click Here ([./blog.md](./blog.md))
---

## Colab link
Click here ([https://colab.research.google.com/drive/1laql7j2QKwFfEXIVlkt1FgHK-lmvr8Qx?usp=sharing](https://colab.research.google.com/drive/1laql7j2QKwFfEXIVlkt1FgHK-lmvr8Qx?usp=sharing))


## Project Structure

```
legacyforge/
├── __init__.py                        # Module exports
├── client.py                          # LegacyforgeEnv client (WebSocket-based)
├── models.py                          # LegacyforgeAction & LegacyforgeObservation
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Project metadata and dependencies
├── Dockerfile                         # Container image
├── uv.lock                            # Locked dependency tree
│
└── server/
    ├── app.py                         # FastAPI server (HTTP + WebSocket)
    ├── legacyforge_env.py             # Core environment logic & reward computation
    ├── legacyforge_environment.py     # OpenEnv Environment interface
    ├── custom_levels.py               # Level definitions
    ├── sandbox.py                     # Sandboxed code execution
    ├── triangle_of_truth.py           # Test validation
    ├── host_cache_builder.py          # Offline docs cache builder
    ├── static_docs_cache.json         # Pre-built documentation cache
    ├── baseline_eval.py               # Rule-based baseline evaluation
    ├── rl_eval.py                     # RL policy evaluation
    ├── generate_training_data.py      # SFT dataset generation
    ├── generate_golden_data.py        # Golden trajectory generation
    ├── sft_dataset.json               # Pre-generated SFT training data
    ├── baseline_4_levels_results.json # Baseline results across 4 levels
    └── levels/
        ├── level1_flask.py            # Level 1 starting Flask code
        ├── level1_tests.py            # Level 1 pytest suite
        ├── level1_answer.py           # Level 1 reference answer
        └── golden_solution_l1.py      # Level 1 golden solution
```
---
