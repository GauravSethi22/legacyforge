# LegacyForge: When Clean Code Meets Real-World Chaos

> *Most coding agents today excel at generating clean code from scratch — but they stumble when faced with the messy reality of existing systems.*

Migrating a legacy Flask application to FastAPI sounds straightforward on paper. In practice, it is a minefield of hidden dependencies, undocumented behaviors, and framework-level gotchas. This gap between ideal conditions and real-world complexity is exactly where **LegacyForge** steps in.

---

## What is LegacyForge?

**LegacyForge** is a self-scaling reinforcement learning (RL) environment where agents migrate legacy code and generate adversarial tests, making the task progressively harder as they improve.

It is not just a benchmark. It is a training ground that grows with the agent.

---

## The Environment

At its core, LegacyForge operates through a structured interaction loop driven by two primitives:

```python
env.reset()
env.step(action)
```

Every episode, the agent encounters a legacy codebase and must navigate its quirks, one deliberate action at a time.

---

## Actions — What the Agent Can Do

The agent interacts with the environment through a well-defined action space:

| Action | Description |
|---|---|
| `read_docs` | Fetch relevant documentation to guide migration decisions |
| `edit_function` | Modify or rewrite parts of the legacy codebase |
| `run_tests` | Execute the test suite to validate the current implementation |
| `code_review` | Analyze code for potential issues or improvements |
| `submit_test` | Propose an adversarial test to challenge the system |

---

## Observations — What the Agent Sees

After every action, the agent receives a rich observation bundle:

- **Legacy code state** — the current version of the evolving codebase
- **Documentation hints** — contextual guidance for framework migration
- **Progress summary** — feedback on actions and migration trajectory
- **Environment state** — phase, test results, and performance metrics

---

## Rewards — What Drives Learning

While observations provide feedback, the reward model drives learning by scoring the agent across multiple dimensions:

### Migration Success
```
+2.0   Successful compilation
+0.3   Per test passed
```

### Adversarial Test Quality
```
+3.0   Valid adversarial test submitted
+0.0 to +2.0   Novelty bonus (penalized if too similar to prior tests)
−2.0   Oracle penalty for invalid or broken test cases
```

### Strategy Quality
```
+1.0   Reads docs before the first edit
+1.0   Runs tests after each edit
−0.5   Random edits made without context
```

> The reward structure is not just about arriving at the right answer — it rewards the discipline of *how* you get there.

---

## Training Setup — Three Levels of Complexity

The agent is trained across three progressively challenging difficulty tiers:

### Level 1 — Single Route
Basic migration with simple bugs: async handling and Pydantic model updates.

### Level 2 — Multi-Route App
Handles multiple endpoints, input validation, and background task coordination.

### Level 3 — Full App Sprint
Real-world complexity: OAuth2 authentication, async database operations, and rate limiting.

---

## Training Setup Pipeline

The pipeline below illustrates the end-to-end flow of the LegacyForge training process — from environment initialization through action execution, reward computation, and policy update. Each episode feeds directly into the next, creating a continuous improvement loop that compounds across all three difficulty levels.

![Training Setup Pipeline](./images/pipeline.png)

*Figure 1: LegacyForge training pipeline — from environment reset to policy convergence across Level 1, 2, and 3 episodes.*

---

## Visualizing RL's Impact on Agent Reasoning

Understanding how reinforcement learning reshapes agent behavior requires looking beyond raw performance numbers. The visualization below captures the decision trajectory of both a baseline agent and an RL-trained agent across a Level 2 migration task, highlighting the divergence in strategy that emerges after sufficient training iterations.

What stands out is not merely the difference in success rates, but the qualitative shift in reasoning patterns. The RL-trained agent develops a consistent pre-edit ritual — consulting documentation, scoping the change, then executing — while the baseline agent jumps directly to edits, often compounding errors rather than resolving them.

![RL Impact Visualization — Baseline Agent](./images/WhatsApp%20Image%202026-04-26%20at%2014.36.31.jpeg)
![RL Impact Visualization — RL-Trained Agent](./images/WhatsApp%20Image%202026-04-26%20at%2014.36.45.jpeg)

*Figure 2: Side-by-side comparison of baseline vs. RL-trained agent decision trajectories on a multi-route migration task.*

| Metric | Baseline Agent | RL-Trained Agent |
|---|---|---|
| Strategy | Trial and error | Stable, learned policy |
| Recovery | Inconsistent | Fast and deliberate |
| Documentation usage | Rarely consulted | Proactively referenced |
| Test quality | Redundant / invalid | Novel and valid |

The baseline model stumbles through migrations inconsistently. The RL-trained agent learns a stable strategy, consults documentation early, and recovers gracefully from mistakes.

---

## What Makes LegacyForge Unique

### Triangle of Truth
Every submitted test must satisfy three properties simultaneously: **solvable**, **correct**, and **genuinely challenging**. This prevents the agent from gaming the reward signal with trivial or redundant tests.

### Oracle Penalty (−2.0)
A strong guardrail against invalid or exploitative test submissions. The agent learns that submitting a broken test carries a greater cost than submitting nothing at all.

### Migration Memory
The environment tracks past errors and surfaces them as part of the agent's observation, enabling iterative improvement grounded in real failure history.

### Dynamic Knowledge Retrieval
Fuzzy search with caching rewards active research over guesswork. Agents that consult documentation perform measurably better — and the reward model reflects that.

---

## Conclusion

LegacyForge does not simply test whether an agent *can* migrate code. It trains agents to migrate code *well* — methodically, evidence-driven, and with a calibrated understanding of what they do not yet know.

> *Real-world codebases are messy. LegacyForge ensures the agents that tackle them are prepared.*
