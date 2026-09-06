"""Report and trace writing utilities for MangoMAS collection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from forge.mangomas.collector.types import ScenarioCollectionResult, _EpisodeRollout


def write_collection_report(
    result: ScenarioCollectionResult,
    output_path: str | Path,
    *,
    mode: str,
    platform: str,
    policy_name: str,
    base_seed: int,
    scenario_refs: Sequence[str | Path],
    config_paths: Sequence[str | Path],
    run_name: str,
) -> Path:
    """Write a JSON report summarizing a MangoMAS collection run."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_report_dict(
        mode=mode,
        platform=platform,
        policy_name=policy_name,
        base_seed=base_seed,
        scenario_refs=scenario_refs,
        config_paths=config_paths,
        run_name=run_name,
    )
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
    return report_path


def _open_trace_writer_if_enabled(teacher_config: Any, scenario_id: str, episode_index: int) -> Any:
    """Open a TeacherTraceWriter when teacher capture + output_root are enabled."""
    if teacher_config is None or not teacher_config.output_root:
        return None
    from forge.mangomas.teacher_trace import TeacherTraceWriter

    return TeacherTraceWriter(
        teacher_config.output_root,
        scenario_id,
        episode_index,
        shard_size=teacher_config.shard_size,
        compress=teacher_config.compress_traces,
    )


def _flush_rollout_to_writer(
    rollout: _EpisodeRollout,
    *,
    scenario_id: str,
    episode_index: int,
    teacher_config: Any,
    writer: Any,
) -> None:
    """Persist an episode rollout's teacher records to ``writer``.

    Used by the async path where trace writing is deferred until episodes
    complete in episode-index order, guaranteeing byte-identical shard
    contents regardless of coroutine completion order.
    """
    from forge.mangomas.teacher_trace import TeacherDecisionTrace

    if rollout.teacher_intentions is None:
        return
    n = len(rollout.action_ids)
    # Use the env-reported action space size when available so trace
    # legal_actions match what the agent could actually pick. Falling back
    # to action_ids.max()+1 underestimates whenever an episode never
    # exercises every legal action.
    legal = (
        list(range(int(rollout.action_space_size)))
        if rollout.action_space_size > 0
        else (list(range(int(rollout.action_ids.max()) + 1)) if n > 0 else [])
    )
    for step_index in range(n):
        intent = rollout.teacher_intentions[step_index]
        writer.log(
            TeacherDecisionTrace(
                scenario_id=scenario_id,
                episode_index=episode_index,
                step_index=step_index,
                observation=dict(rollout.raw_observations[step_index])
                if step_index < len(rollout.raw_observations)
                else {},
                legal_actions=legal,
                action_id=int(rollout.action_ids[step_index]),
                intention=intent if intent >= 0 else None,
                subgoals=(rollout.teacher_subgoals or [])[step_index]
                if rollout.teacher_subgoals
                else [],
                rationale=(rollout.teacher_rationales or [])[step_index]
                if rollout.teacher_rationales
                else "",
                value_hat=(rollout.teacher_value_hats or [])[step_index]
                if rollout.teacher_value_hats
                else 0.0,
                constraint_critique=(rollout.teacher_constraint_critiques or [])[step_index]
                if rollout.teacher_constraint_critiques
                else {},
                top_k_probs=(rollout.teacher_top_k_probs or [])[step_index]
                if rollout.teacher_top_k_probs
                else [],
                provider=(
                    (rollout.teacher_providers or [teacher_config.provider])[step_index]
                    if rollout.teacher_providers and step_index < len(rollout.teacher_providers)
                    else teacher_config.provider
                ),
                model=teacher_config.model,
                prompt_tokens=(
                    rollout.teacher_prompt_tokens[step_index]
                    if rollout.teacher_prompt_tokens
                    and step_index < len(rollout.teacher_prompt_tokens)
                    else 0
                ),
                completion_tokens=(
                    rollout.teacher_completion_tokens[step_index]
                    if rollout.teacher_completion_tokens
                    and step_index < len(rollout.teacher_completion_tokens)
                    else 0
                ),
                latency_ms=(
                    rollout.teacher_latency_ms[step_index]
                    if rollout.teacher_latency_ms and step_index < len(rollout.teacher_latency_ms)
                    else 0.0
                ),
                schema_version=teacher_config.trace_schema_version,
            )
        )
