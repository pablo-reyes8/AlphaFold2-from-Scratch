# AlphaFold2: A PyTorch Reconstruction

<div align="center">

**Dissecting geometric deep learning and structural biology representations from scratch.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](#installation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Status](https://img.shields.io/badge/status-Research%20Prototype-orange)](#project-status)

</div>

> **🚧 Development Status:** This repository is a living research environment under active development. We are currently iterating on core architectural modules, optimizing the geometric forward pass, and expanding our structural validation suite. Updates are frequent as we refine the implementation toward full end-to-end parity with the AlphaFold2 design space.

---

## Overview

This repository provides a **from-scratch, modular PyTorch implementation of the core AlphaFold2 architecture**. 

While the original DeepMind release and frameworks like OpenFold are built for large-scale production, this project is engineered for **architectural transparency and research experimentation**. It breaks down the monolithic structural biology pipeline into inspectable, hackable modules, allowing researchers to study exactly how Multiple Sequence Alignments (MSA), pair representations, and geometric heads interact at the tensor level.

## Architectural Focus

The implementation strictly follows the representational flow of the original paper, providing clean PyTorch modules for:

* **Representational Flow:** Explicit handling of MSA, Pair, and Single state embeddings.
* **The Evoformer:** Fully implemented axial attention mechanisms and triangle updates for spatial reasoning.
* **Structure Module:** Native PyTorch implementations of **Invariant Point Attention (IPA)**, rigid body transformations, and structural loss computations (FAPE).
* **Geometric Precision:** Robust unit testing suite specifically targeting structural losses and rotational invariants.

## Data & Reproducibility

Instead of relying on opaque data pipelines, this repository enforces a **manifest-driven workflow**. By decoupling the dataloader from raw folder structures, experiments become inherently more reproducible.

* **Foldbench Integration:** Includes scripts to pull and preprocess a subset of Foldbench.
* **Config-Driven:** Fully parameterizable experiments via YAML (model size, depth, learning rates, EMA).
* **Inspection Tooling:** CLI utilities to sanity-check manifests, A3M files, and CA distance maps before initiating training loops.

---

## Repository structure

```text
.
├── config/
│   ├── data/
│   │   └── foldbench_subset.yaml
│   └── experiments/
│       ├── af2_poc.yaml
│       └── alphafold2_full_reference.yaml
├── data/
│   ├── download_data.sh
│   ├── preproces_data.py
│   ├── dataloaders.py
│   ├── visualize_data.py
│   └── Proteinas_secuencias.csv
├── scripts/
│   ├── prepare_data.py
│   ├── inspect_data.py
│   ├── validate_model.py
│   └── train_model.py
├── model/
├── training/
├── tests/
├── notebooks/
├── requirements.txt
├── Dockerfile
└── README.md
```

### Key files

- `data/download_data.sh` — downloads the Foldbench subset using a target file or CSV input.
- `data/preproces_data.py` — builds or rewrites the manifest and emits YAML summaries.
- `data/dataloaders.py` — dataset code supporting both manifest-based and raw-folder loading.
- `data/visualize_data.py` — command-line inspection utilities for manifests, A3M previews, and CA distance maps.
- `scripts/prepare_data.py` — high-level CLI to download data, refresh manifests, and smoke-test dataloaders.
- `scripts/inspect_data.py` — concise dataset inspection CLI, including batch previews and a simple 3D backbone renderer.
- `scripts/validate_model.py` — instantiates the model stack, runs synthetic forward validation, and dispatches pytest.
- `scripts/train_model.py` — config-driven launcher for end-to-end training runs.
- `config/experiments/af2_poc.yaml` — lightweight proof-of-concept experiment config.
- `config/experiments/alphafold2_full_reference.yaml` — reference values collected from AlphaFold/OpenFold-style configs.

---

## Quickstart

### 1) Create an environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Download the subset

```bash
bash data/download_data.sh --targets-csv data/Proteinas_secuencias.csv
```

Or through the repo CLI:

```bash
python3 scripts/prepare_data.py download --targets-csv data/Proteinas_secuencias.csv
```

### 3) Rebuild the manifest with local paths

```bash
python3 -m data.preproces_data \
  --config config/data/foldbench_subset.yaml \
  --json-path data/af_subset/jsons/fb_protein.json \
  --msa-root data/af_subset/foldbench_msas \
  --cif-root data/af_subset/reference_structures
```

### 4) Inspect the dataset

```bash
python3 -m data.visualize_data manifest-summary --manifest-csv data/Proteinas_secuencias.csv
python3 -m data.visualize_data msa-preview --a3m-path data/af_subset/foldbench_msas/7qrj_A/cfdb_hits.a3m
```

Or use the higher-level inspection scripts:

```bash
python3 scripts/inspect_data.py loader-preview --config config/experiments/af2_poc.yaml --max-samples 2
python3 scripts/inspect_data.py protein-3d \
  --cif-path data/af_subset/reference_structures/7qrj-assembly1_1.cif \
  --chain-id A \
  --output artifacts/7qrj_A_backbone.png
```

### 5) Use the manifest in the dataset

```python
from data.dataloaders import FoldbenchProteinDataset

dataset = FoldbenchProteinDataset(manifest_csv="data/Proteinas_secuencias.csv")
```

---

## Bundled Showcase Data

The repository now includes a tiny real-data showcase under `data/af_subset_showcase` plus:

- `data/showcase_manifest.csv`
- `data/showcase_manifest.yaml`
- `data/showcase_targets.txt`

Current showcase targets:

- `7qrj_A`
- `7qrr_A`

This showcase subset is intentionally small, around `2.2M` on disk, so it can be versioned in the repository as a reproducible sanity-check for the data pipeline without dragging in a large benchmark snapshot.

You can validate the real local data path with:

```bash
python3 scripts/prepare_data.py loader-smoke \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --batch-size 2 \
  --max-samples 2
```

And inspect one of the bundled structures with:

```bash
python3 scripts/inspect_data.py protein-3d \
  --cif-path data/af_subset_showcase/reference_structures/7qrj-assembly1_68.cif \
  --chain-id A \
  --output artifacts/showcase_7qrj_backbone.png
```



---

## CLI workflows

### Prepare data and loader

```bash
python3 scripts/prepare_data.py bootstrap \
  --data-config config/data/foldbench_subset.yaml \
  --experiment-config config/experiments/af2_poc.yaml
```

### Validate the model stack

```bash
python3 scripts/validate_model.py instantiate --config config/experiments/af2_poc.yaml
python3 scripts/validate_model.py forward-smoke --config config/experiments/af2_poc.yaml --device cpu
python3 scripts/validate_model.py pytest --target tests --pytest-arg=-q
```

### Run a safe training smoke test

```bash
python3 scripts/train_model.py --config config/experiments/af2_poc.yaml --device cpu --dry-run
```

### Launch training with recycling overrides

```bash
python3 scripts/train_model.py \
  --config config/experiments/af2_poc.yaml \
  --device cuda \
  --num-recycles 0 \
  --stochastic-recycling \
  --max-recycles 3
```

---

## Configs

### `config/experiments/af2_poc.yaml`

This config mirrors the current notebook-scale proof of concept and is suitable for smaller experimental runs.

Current example values:

- `max_msa_seqs: 128`
- `batch_size: 2`
- `epochs: 20`
- `lr: 1e-4`
- `num_evoformer_blocks: 2`
- `num_structure_blocks: 4`

### `config/experiments/alphafold2_full_reference.yaml`

This file is a **reference document**, not a statement that the current code already consumes every field end-to-end.

Its role is to provide a structured target for future extension and to document the broader AlphaFold/OpenFold design space.

---

## Docker

A small CPU-oriented image can be built with:

```bash
docker build -t alphafold-from-scratch .
```

This image is intended for **environment setup, utilities, and data tooling**, not for serious GPU training.

---

## Design Philosophy

This repository is architected with a singular premise: **true understanding of geometric deep learning requires unconstrained access to its atomic components.**

Rather than providing a monolithic black box or a superficial tutorial, this codebase is engineered specifically for deep architectural study and rapid ablation. It strips away the distributed production overhead of frameworks like OpenFold to expose the bare mathematical and algorithmic reality of the network.

**Core Principles:**

* **Architectural Transparency:** Designed to be read, debugged, and mathematically verified at the tensor level. There is no hidden logic; the mapping from the original paper's equations to PyTorch modules is direct and explicit.
* **Modular Extensibility:** Every mechanism—from the Evoformer's axial attention to the Invariant Point Attention (IPA)—is fully decoupled. Researchers can isolate, modify, or completely redesign structural modules without fighting the framework.
* **Rigorous Prototyping:** Provides a robust, high-fidelity environment for testing novel geometric learning hypotheses, custom attention mechanisms, and alternative structural losses before scaling them to production clusters.

This makes the repository a specialized tool for researchers dissecting structural biology models, engineers debugging complex 3D equivariance, and anyone focused on advancing the theoretical foundations of the AlphaFold family.

---

## Intended audience

This project may be useful for:

- ML researchers studying geometric deep learning or protein structure prediction,
- students implementing AlphaFold2-style systems to truly understand them,
- engineers who want a smaller environment for experimentation,
- researchers building derivatives, ablations, or teaching materials.

It is probably **not** the best starting point if your main goal is immediately obtaining state-of-the-art folding performance with industrial robustness. In that case, official or mature large-scale implementations will usually be a better operational choice.


---

## Roadmap

A realistic roadmap for this repository could include:

- [ ] tighter end-to-end training validation
- [ ] expanded benchmark and evaluation scripts
- [ ] example inference notebook or script
- [ ] reproducibility report for a reference training run
- [ ] Visualizations for understanding AlphaFold

---

## Citation

If this repository helps your work, please cite the repository and also cite the original AlphaFold and related implementation papers that inspired the architecture.

A simple placeholder BibTeX entry for the repository could look like:

```bibtex
@software{reyes_alphafold2_from_scratch,
  author = {Pablo Reyes},
  title = {AlphaFold2 From Scratch},
  year = {2026},
  url = {https://github.com/pablo-reyes8/alpha-fold2}
}
```

---

## Acknowledgments

This repository is inspired by the AlphaFold2 line of work and the broader ecosystem of open implementations and educational reverse-engineering efforts around protein structure prediction.

Special credit belongs to the original AlphaFold work and to the open-source community that has made this field far more accessible to study.

---

## License

This project is licensed under the **MIT License**.
