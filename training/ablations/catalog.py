"""Registry of prebuilt ablation presets for the AlphaFold2 project.

Each preset is intentionally high-level: it toggles top-level wiring or loss
weights while leaving the internal Evoformer, IPA, and structure mathematics
unchanged. The goal is to make comparative training runs easy to launch from
the CLI and safe to maintain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    """Describe a named ablation and the config overrides needed to activate it."""

    key: str
    title: str
    category: str
    description: str
    overrides: dict[str, Any]


_ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec(
        key="AF2_1",
        title="Purely Evolutionary Trunk",
        category="architecture",
        description=(
            "Disables the Evoformer pair stack, disables recycling, and removes "
            "the pLDDT auxiliary so z is updated only through the MSA-to-pair bridge."
        ),
        overrides={
            "model": {"ablation": 1},
            "loss": {"ablation": 1},
            "trainer": {
                "num_recycles": 0,
                "stochastic_recycling": False,
                "max_recycles": 0,
            },
        },
    ),
    AblationSpec(
        key="AF2_2",
        title="Triangle Multiplication Only",
        category="architecture",
        description=(
            "Keeps triangle multiplication active, disables triangle attention, "
            "disables recycling, and removes the pLDDT auxiliary."
        ),
        overrides={
            "model": {"ablation": 2},
            "loss": {"ablation": 2},
            "trainer": {
                "num_recycles": 0,
                "stochastic_recycling": False,
                "max_recycles": 0,
            },
        },
    ),
    AblationSpec(
        key="AF2_3",
        title="FAPE Only",
        category="loss",
        description=(
            "Removes distogram, pLDDT, and torsion auxiliaries so optimization "
            "is driven only by FAPE."
        ),
        overrides={
            "model": {"ablation": 3},
            "loss": {"ablation": 3},
        },
    ),
    AblationSpec(
        key="AF2_4",
        title="Untied Structure Module",
        category="architecture",
        description=(
            "Turns on block-specific parameters in the structure module so each "
            "structure block can learn its own update rule."
        ),
        overrides={"model": {"ablation": 4}},
    ),
    AblationSpec(
        key="AF2_5",
        title="No Evoformer Relational Trunk",
        category="architecture",
        description=(
            "Bypasses the Evoformer entirely and feeds raw input embeddings into "
            "the later stages, with recycling disabled."
        ),
        overrides={
            "model": {"ablation": 5},
            "trainer": {
                "num_recycles": 0,
                "stochastic_recycling": False,
                "max_recycles": 0,
            },
        },
    ),
)

_ABLATION_BY_KEY = {spec.key.upper(): spec for spec in _ABLATION_SPECS}


def list_ablation_specs() -> list[AblationSpec]:
    """Return the ablation registry in declaration order."""

    return list(_ABLATION_SPECS)


def get_ablation_spec(name: str) -> AblationSpec:
    """Return a specific ablation spec by case-insensitive key."""

    key = str(name).strip().upper()
    if key not in _ABLATION_BY_KEY:
        valid = ", ".join(spec.key for spec in _ABLATION_SPECS)
        raise KeyError(f"Unknown ablation '{name}'. Available ablations: {valid}")
    return _ABLATION_BY_KEY[key]
