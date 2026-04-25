from typing import Literal, Optional, Dict, Any
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

class LegacyforgeAction(Action):
    action_type: Literal["read_docs", "edit_function", "run_tests", "code_review", "submit_test"] = Field(..., description="Type of action to perform")
    target: Optional[str] = Field(default=None, description="Target component or function")
    code: Optional[str] = Field(default=None, description="Code content for editing")

class LegacyforgeObservation(Observation):
    legacy_code: str = Field(default="", description="The legacy codebase state")
    docs: str = Field(default="", description="Documentation")
    migration_history_summary: str = Field(default="", description="History of migrations performed")
    level: int = Field(default=1, description="Current environment level")
    
    observation: Optional[Dict[str, Any]] = Field(default=None, description="Nested observation dict")
    reward_breakdown: Optional[Dict[str, Any]] = Field(default=None, description="Breakdown of rewards")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Additional info")
