# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Legacyforge Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import LegacyforgeAction, LegacyforgeObservation


class LegacyforgeEnv(
    EnvClient[LegacyforgeAction, LegacyforgeObservation, State]
):
    """
    Client for the Legacyforge Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with LegacyforgeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(LegacyforgeAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = LegacyforgeEnv.from_docker_image("legacyforge-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(LegacyforgeAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: LegacyforgeAction) -> Dict:
        """
        Convert LegacyforgeAction to JSON payload for step message.

        Args:
            action: LegacyforgeAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type,
            "target": action.target,
            "code": action.code,
        }

    def _parse_result(self, payload: Dict) -> StepResult[LegacyforgeObservation]:
        """
        Parse server response into StepResult[LegacyforgeObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with LegacyforgeObservation
        """
        obs_data = payload.get("observation", {})
        observation = LegacyforgeObservation(
            legacy_code=obs_data.get("legacy_code", ""),
            docs=obs_data.get("docs", ""),
            migration_history_summary=obs_data.get("migration_history_summary", ""),
            level=obs_data.get("level", 1),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            info=payload.get("info", {}),
            reward_breakdown=payload.get("reward_breakdown", {})
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
