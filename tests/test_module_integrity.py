"""Verify that key project modules import cleanly and expose nontrivial module docstrings."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


MODULES = [
    "model.alphafold2",
    "model.alphafold2_full_loss",
    "model.alphafold2_heads",
    "model.evoformer_block",
    "model.evoformer_stack",
    "model.input_embedder",
    "model.invariant_point_attention",
    "model.ipa_transformations",
    "model.losses.distogram_loss",
    "model.losses.fape_loss",
    "model.losses.loss_helpers",
    "model.losses.pLDDT_loss",
    "model.losses.structure_aux_loss",
    "model.losses.torsion_loss",
    "model.msa_colum_attention",
    "model.msa_row_attention",
    "model.msa_transitions",
    "model.outer_product_mean",
    "model.quaterion_to_matrix",
    "model.structure_block",
    "model.structure_transation",
    "model.torsion_head",
    "model.triange_attention",
    "model.triangle_multiplication",
    "training.autocast",
    "training.chekpoints",
    "training.colab_utils",
    "training.efficent_metrics",
    "training.ema",
    "training.metrics_for_alphafold",
    "training.metrics_utils",
    "training.scheduler_warmup",
    "training.seeds",
    "training.train_alphafold2",
    "training.train_one_epoch",
]

MODULE_PATHS = sorted(Path("model").rglob("*.py")) + sorted(Path("training").rglob("*.py"))


@pytest.mark.parametrize("module_name", MODULES)
def test_project_modules_import_cleanly(module_name):
    module = importlib.import_module(module_name)
    assert module is not None


@pytest.mark.parametrize("path", MODULE_PATHS, ids=lambda path: str(path))
def test_project_modules_have_descriptive_module_docstrings(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring = ast.get_docstring(tree)
    assert docstring, f"Missing module docstring in {path}"
    assert len(docstring.split()) >= 12, f"Module docstring too short in {path}"
