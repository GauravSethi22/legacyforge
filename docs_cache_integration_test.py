"""
docs_cache_integration_test.py

Tests the real offline docs cache integration inside LegacyforgeEnvironment:
  1. Exact cache key match    (tutorial/first-steps)
  2. Fuzzy match              (pydantic -> closest page)
  3. Cache miss               (total garbage topic)
  4. Hardcoded fallback       (async -> DOCS_SNIPPETS, simulated by wiping cache)
  5. Source tag is always present in info
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

from server.legacyforge_env import LegacyforgeEnvironment
from models import LegacyforgeAction

print("=" * 60)
print("DOCS CACHE INTEGRATION TEST")
print("=" * 60)

env = LegacyforgeEnvironment()
print(f"\nCache loaded: {len(env._docs_cache)} entries\n")

# ── Test 1: Exact cache key ──────────────────────────────────────────────────
obs = env.step(LegacyforgeAction(action_type="read_docs", target="tutorial/first-steps"))
source = obs.info.get("source")
snippet = obs.info.get("docs_snippet", "")
assert source == "cache_exact", f"Expected cache_exact, got {source}"
assert "FastAPI" in snippet or "app" in snippet, "Snippet looks empty"
print("Test 1 PASSED — Exact match (tutorial/first-steps)")
print(f"  source : {source}")
print(f"  preview: {snippet[:120].strip()!r}\n")

# ── Test 2: Fuzzy match ───────────────────────────────────────────────────────
obs = env.step(LegacyforgeAction(action_type="read_docs", target="pydantic"))
source = obs.info.get("source")
snippet = obs.info.get("docs_snippet", "")
assert source == "cache_fuzzy", f"Expected cache_fuzzy, got {source}"
assert "[Redirected" in snippet, "Missing redirect prefix"
print("Test 2 PASSED — Fuzzy match (pydantic)")
print(f"  source : {source}")
print(f"  preview: {snippet[:120].strip()!r}\n")

# ── Test 3: Cache miss (garbage topic) ───────────────────────────────────────
obs = env.step(LegacyforgeAction(action_type="read_docs", target="xyzzy_no_such_topic_999"))
source = obs.info.get("source")
snippet = obs.info.get("docs_snippet", "")
assert source == "cache_miss", f"Expected cache_miss, got {source}"
assert "not found" in snippet.lower() or "Error" in snippet, "Missing error message"
print("Test 3 PASSED -- Cache miss (garbage topic)")
print(f"  source : {source}")
print(f"  preview: {snippet[:120].strip()!r}\n")

# ── Test 4: Hardcoded fallback when cache is empty ───────────────────────────
env2 = LegacyforgeEnvironment()
env2._docs_cache = {}   # simulate missing static_docs_cache.json
obs = env2.step(LegacyforgeAction(action_type="read_docs", target="async"))
source = obs.info.get("source")
snippet = obs.info.get("docs_snippet", "")
assert source == "hardcoded", f"Expected hardcoded, got {source}"
assert "async def" in snippet, "Hardcoded snippet should contain 'async def'"
print("Test 4 PASSED -- Hardcoded fallback (cache wiped, topic=async)")
print(f"  source : {source}")
print(f"  preview: {snippet[:120].strip()!r}\n")

# ── Test 5: Source tag is always present ─────────────────────────────────────
for topic in ["tutorial/path-params", "exceptions", "routing", "abc123"]:
    o = env.step(LegacyforgeAction(action_type="read_docs", target=topic))
    assert "source" in o.info, f"source key missing for topic={topic}"
print("Test 5 PASSED -- source tag present for all topics\n")

print("=" * 60)
print("ALL DOCS CACHE TESTS PASSED")
print("=" * 60)
