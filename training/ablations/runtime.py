"""Runtime helpers for resolving ablation presets into concrete configs.

The utilities in this module deep-merge a named ablation onto an existing YAML
experiment config, annotate metadata, and standardize run naming so ablation
jobs can be launched from either the single-device or multi-GPU training CLIs.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from training.ablations.catalog import AblationSpec, get_ablation_spec, list_ablation_specs


def _deep_merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "ablation"


def build_baseline_spec() -> AblationSpec:
    """Return a synthetic spec describing the non-ablated baseline."""

    return AblationSpec(
        key="BASELINE",
        title="Baseline",
        category="baseline",
        description="No ablation overrides. Uses the base experiment config as-is.",
        overrides={},
    )


def apply_ablation_overrides(
    base_config: dict[str, Any],
    *,
    spec: AblationSpec,
) -> dict[str, Any]:
    """Apply one ablation spec to a base config and annotate the result."""

    resolved = _deep_merge_dicts(base_config, spec.overrides)

    metadata = dict(resolved.get("metadata", {}) or {})
    base_name = str(metadata.get("name", "alphafold2"))
    slug = _slugify(f"{spec.key}_{spec.title}")
    metadata["base_config_name"] = base_name
    metadata["name"] = f"{base_name}_{slug}"
    metadata["ablation_id"] = spec.key
    metadata["ablation_title"] = spec.title
    metadata["ablation_category"] = spec.category
    metadata["ablation_description"] = spec.description
    resolved["metadata"] = metadata

    trainer = dict(resolved.get("trainer", {}) or {})
    base_run_name = str(trainer.get("run_name", base_name))
    base_ckpt_dir = str(trainer.get("ckpt_dir", "checkpoints_af2"))
    trainer["run_name"] = f"{base_run_name}_{slug}"
    trainer["ckpt_dir"] = f"{base_ckpt_dir}/ablations/{slug}"
    resolved["trainer"] = trainer

    return resolved


def apply_ablation_modifiers(
    config: dict[str, Any],
    *,
    single_sequence_msa: bool = False,
    use_block_specific_params: bool | None = None,
) -> dict[str, Any]:
    """Apply orthogonal data or structure modifiers on top of a resolved variant."""

    resolved = deepcopy(config)
    suffixes: list[str] = []

    if single_sequence_msa:
        data_cfg = dict(resolved.get("data", {}) or {})
        data_cfg["single_sequence_mode"] = True
        data_cfg["max_msa_seqs"] = 1
        resolved["data"] = data_cfg
        suffixes.append("single_sequence")

    if use_block_specific_params is not None:
        model_cfg = dict(resolved.get("model", {}) or {})
        model_cfg["use_block_specific_params"] = bool(use_block_specific_params)
        resolved["model"] = model_cfg
        suffixes.append("structure_untied" if use_block_specific_params else "structure_shared")

    if suffixes:
        suffix = "_".join(suffixes)
        metadata = dict(resolved.get("metadata", {}) or {})
        trainer = dict(resolved.get("trainer", {}) or {})
        metadata["name"] = f"{metadata.get('name', 'alphafold2')}_{suffix}"
        trainer["run_name"] = f"{trainer.get('run_name', metadata.get('name', 'alphafold2'))}_{suffix}"
        trainer["ckpt_dir"] = f"{trainer.get('ckpt_dir', 'checkpoints_af2')}/{suffix}"
        resolved["metadata"] = metadata
        resolved["trainer"] = trainer

    return resolved


def resolve_ablation_config(
    base_config: dict[str, Any],
    *,
    ablation_name: str,
) -> tuple[dict[str, Any], AblationSpec]:
    """Resolve a named ablation into a concrete config and return its spec."""

    spec = get_ablation_spec(ablation_name)
    return apply_ablation_overrides(base_config, spec=spec), spec


def resolve_training_variant(
    base_config: dict[str, Any],
    *,
    ablation_name: str | None = None,
    single_sequence_msa: bool = False,
    use_block_specific_params: bool | None = None,
) -> tuple[dict[str, Any], AblationSpec]:
    """Resolve the baseline or an ablation preset and apply orthogonal modifiers."""

    if ablation_name is None or str(ablation_name).strip().upper() in {"BASELINE", "NONE"}:
        spec = build_baseline_spec()
        config = apply_ablation_overrides(base_config, spec=spec)
    else:
        config, spec = resolve_ablation_config(base_config, ablation_name=ablation_name)

    config = apply_ablation_modifiers(
        config,
        single_sequence_msa=single_sequence_msa,
        use_block_specific_params=use_block_specific_params,
    )
    return config, spec


def render_ablation_catalog() -> str:
    """Format the ablation registry as a small human-readable table."""

    lines = [
        "Available ablations:",
        "- BASELINE [baseline] Baseline: No ablation overrides. Uses the base config as-is.",
    ]
    for spec in list_ablation_specs():
        lines.append(f"- {spec.key} [{spec.category}] {spec.title}: {spec.description}")
    lines.extend(
        [
            "",
            "Orthogonal modifiers:",
            "- single_sequence_msa: collapse the MSA to the target sequence only.",
            "- use_block_specific_params: untie the parameters across structure blocks.",
        ]
    )
    return "\n".join(lines)
