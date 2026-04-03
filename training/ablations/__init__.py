"""Config-driven ablation presets for the AlphaFold2 training stack.

This package exposes a small registry of architecture and loss ablations that
can be applied on top of an existing experiment config without changing the
core training loop or rewriting the internal model blocks.
"""

from training.ablations.catalog import AblationSpec, get_ablation_spec, list_ablation_specs
from training.ablations.runtime import (
    apply_ablation_modifiers,
    apply_ablation_overrides,
    build_baseline_spec,
    render_ablation_catalog,
    resolve_ablation_config,
    resolve_training_variant,
)

__all__ = [
    "AblationSpec",
    "apply_ablation_modifiers",
    "apply_ablation_overrides",
    "build_baseline_spec",
    "get_ablation_spec",
    "list_ablation_specs",
    "render_ablation_catalog",
    "resolve_ablation_config",
    "resolve_training_variant",
]
