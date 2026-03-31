from __future__ import annotations

import pytest
import torch

from model.alphafold2 import AlphaFold2
from model.alphafold2_full_loss import AlphaFoldLoss


def _random_unit_vectors(shape):
    values = torch.randn(*shape)
    return values / torch.linalg.norm(values, dim=-1, keepdim=True).clamp_min(1e-8)


@pytest.fixture
def ideal_backbone_local():
    return torch.tensor(
        [
            [-1.458, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.547, 1.426, 0.0],
            [0.224, 2.617, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def toy_batch(ideal_backbone_local):
    torch.manual_seed(123)

    batch_size, msa_depth, length, n_torsions = 2, 3, 6, 3
    seq_tokens = torch.randint(1, 27, (batch_size, length))
    msa_tokens = torch.randint(1, 27, (batch_size, msa_depth, length))

    seq_mask = torch.ones(batch_size, length, dtype=torch.float32)
    seq_mask[0, -1] = 0.0
    msa_mask = seq_mask[:, None, :].repeat(1, msa_depth, 1)

    residue_axis = torch.arange(length, dtype=torch.float32)
    coords_ca = torch.stack(
        [residue_axis, 0.1 * residue_axis, torch.zeros_like(residue_axis)],
        dim=-1,
    )
    coords_ca = coords_ca.unsqueeze(0).repeat(batch_size, 1, 1)
    coords_ca = coords_ca + 0.01 * torch.randn_like(coords_ca)

    coords_n = coords_ca + torch.tensor([-1.2, 0.4, 0.1], dtype=torch.float32)
    coords_c = coords_ca + torch.tensor([1.3, 0.5, -0.1], dtype=torch.float32)

    valid_res_mask = seq_mask.clone()
    valid_backbone_mask = seq_mask.clone()

    torsion_true = _random_unit_vectors((batch_size, length, n_torsions, 2)).to(torch.float32)
    torsion_mask = valid_backbone_mask.unsqueeze(-1).expand(batch_size, length, n_torsions).clone()

    pair_mask = valid_res_mask[:, :, None] * valid_res_mask[:, None, :]

    return {
        "seq_tokens": seq_tokens,
        "msa_tokens": msa_tokens,
        "seq_mask": seq_mask,
        "msa_mask": msa_mask,
        "coords_n": coords_n,
        "coords_ca": coords_ca,
        "coords_c": coords_c,
        "valid_res_mask": valid_res_mask,
        "valid_backbone_mask": valid_backbone_mask,
        "pair_mask": pair_mask,
        "torsion_true": torsion_true,
        "torsion_mask": torsion_mask,
        "ideal_backbone_local": ideal_backbone_local,
    }


@pytest.fixture
def toy_model():
    torch.manual_seed(7)
    model = AlphaFold2(
        n_tokens=27,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=0,
        num_evoformer_blocks=1,
        num_structure_blocks=1,
        transition_expansion_evoformer=2,
        transition_expansion_structure=2,
        use_block_specific_params=False,
        dist_bins=64,
        plddt_bins=50,
        n_torsions=3,
        num_res_blocks_torsion=1,
    )
    model.eval()
    return model


@pytest.fixture
def toy_criterion():
    return AlphaFoldLoss(
        fape_length_scale=10.0,
        fape_clamp_distance=10.0,
        dist_num_bins=64,
        dist_min_bin=2.0,
        dist_max_bin=22.0,
        plddt_num_bins=50,
        plddt_inclusion_radius=15.0,
        w_fape=0.5,
        w_dist=0.3,
        w_plddt=0.01,
        w_torsion=0.01,
    )
