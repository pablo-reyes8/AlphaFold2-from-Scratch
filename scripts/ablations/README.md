# Ablation Suite

This folder documents the repository's ablation workflow and the reasoning behind each predefined variant.

## Safety Guarantee

The default training path remains unchanged.

If you instantiate:

- `AlphaFold2(...)`
- `AlphaFoldLoss(...)`

without an explicit ablation id, the model and loss stay in their baseline configuration. All ablation behavior is opt-in and activated only through the dedicated ablation scripts or through explicit `ablation=` arguments.

## Predefined Presets

### AF2_1: Purely Evolutionary Trunk

Hypothesis:
MSA coevolution alone may not be sufficient once the geometric pair stack is removed.

What changes:

- disables the Evoformer pair stack
- keeps only the `OuterProductMean` bridge from MSA to pair
- disables recycling
- disables the pLDDT auxiliary head and loss

### AF2_2: Triangle Multiplication Only

Hypothesis:
Triangle multiplication may retain useful 2D relational structure even if triangle attention is removed.

What changes:

- disables triangle attention in the Evoformer
- keeps triangle multiplication active
- keeps pair transition active
- disables recycling
- disables the pLDDT auxiliary head and loss

### AF2_3: FAPE Only

Hypothesis:
Without auxiliary scaffolding losses, optimization may become much harder even if the structural objective is still correct.

What changes:

- removes distogram loss
- removes pLDDT loss
- removes torsion loss
- disables the corresponding auxiliary heads

### AF2_4: Untied Structure Module

Hypothesis:
The structure stack may benefit from block-specific parameters instead of repeatedly applying the same transition rule.

What changes:

- turns `use_block_specific_params=True` in the structure module

### AF2_5: No Evoformer Relational Trunk

Hypothesis:
The IPA-based structure stage should degrade sharply when the relational trunk is bypassed.

What changes:

- bypasses the Evoformer entirely
- feeds raw input-embedder outputs into later stages
- disables recycling

## Orthogonal Modifiers

These can be combined with the baseline or with any preset.

### `--single-sequence-msa`

This is the repository's no-MSA / zero-shot data ablation.

What it does:

- collapses the MSA to the target sequence only
- sets `max_msa_seqs=1`
- leaves the rest of the data pipeline intact

### `--use-block-specific-params`

This forces untied structure-module parameters even if the chosen preset did not request them.

## Main Entry Points

- `scripts/train_ablation.py`
- `scripts/train_ablation_parallel.py`
- `scripts/ablations/run_suite.py`

## Typical Commands

List presets:

```bash
python3 scripts/train_ablation.py --list
```

Inspect a resolved config without training:

```bash
python3 scripts/train_ablation.py \
  --config config/experiments/af2_poc.yaml \
  --ablation AF2_1 \
  --show
```

Inspect the same preset with the single-sequence modifier:

```bash
python3 scripts/train_ablation.py \
  --config config/experiments/af2_poc.yaml \
  --ablation AF2_1 \
  --single-sequence-msa \
  --show
```

Train one preset:

```bash
python3 scripts/train_ablation.py \
  --config config/experiments/af2_poc.yaml \
  --ablation AF2_3 \
  --device cuda
```

Train a preset with multi-GPU parallelism:

```bash
torchrun --nproc_per_node=2 scripts/train_ablation_parallel.py \
  --config config/experiments/af2_poc.yaml \
  --manifest-csv data/showcase_manifest.csv \
  --ablation AF2_5 \
  --parallel-mode ddp
```

Run a whole sweep and export a comparison table:

```bash
python3 scripts/ablations/run_suite.py \
  --config config/experiments/af2_poc.yaml \
  --include-baseline \
  --all \
  --output-dir artifacts/ablation_suite
```
