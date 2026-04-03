"""Smoke tests for optional pre-Evoformer frontend modules."""

from __future__ import annotations

import torch

from model.extra_msa_stack import ExtraMsaStack
from model.recycling_module import RecyclingEmbedder
from model.template_stack import TemplateStack


def test_recycling_embedder_smoke():
    torch.manual_seed(5)

    B, N, L, c_m, c_z = 2, 3, 6, 256, 128
    module = RecyclingEmbedder(c_m=c_m, c_z=c_z)

    m = torch.randn(B, N, L, c_m)
    z = torch.randn(B, L, L, c_z)
    prev_m1 = torch.randn(B, L, c_m)
    prev_z = torch.randn(B, L, L, c_z)
    prev_positions = torch.randn(B, L, 3)
    seq_mask = torch.ones(B, L, dtype=torch.float32)
    msa_mask = seq_mask[:, None, :].expand(B, N, L).clone()

    m_out, z_out = module(
        m,
        z,
        prev_m1=prev_m1,
        prev_z=prev_z,
        prev_positions=prev_positions,
        seq_mask=seq_mask,
        msa_mask=msa_mask,
    )

    assert m_out.shape == m.shape
    assert z_out.shape == z.shape
    assert torch.isfinite(m_out).all()
    assert torch.isfinite(z_out).all()
    assert not torch.allclose(m_out, m)
    assert not torch.allclose(z_out, z)


def test_template_stack_smoke():
    torch.manual_seed(6)

    B, N, T, L = 2, 3, 2, 6
    c_m, c_z = 256, 128
    module = TemplateStack(c_m=c_m, c_z=c_z)

    m = torch.randn(B, N, L, c_m)
    z = torch.randn(B, L, L, c_z)
    template_angle_feat = torch.randn(B, T, L, 51)
    template_pair_feat = torch.randn(B, T, L, L, 88)
    template_mask = torch.ones(B, T, L, dtype=torch.float32)
    template_mask[:, :, -1] = 0.0

    m_out, z_out = module(
        m,
        z,
        template_angle_feat=template_angle_feat,
        template_pair_feat=template_pair_feat,
        template_mask=template_mask,
    )

    assert m_out.shape == (B, N + T, L, c_m)
    assert z_out.shape == z.shape
    assert torch.isfinite(m_out).all()
    assert torch.isfinite(z_out).all()


def test_extra_msa_stack_smoke():
    torch.manual_seed(7)

    B, N, E, L = 2, 3, 4, 6
    c_m, c_z = 256, 128
    module = ExtraMsaStack(c_m=c_m, c_z=c_z)

    m = torch.randn(B, N, L, c_m)
    z = torch.randn(B, L, L, c_z)
    extra_msa_feat = torch.randn(B, E, L, 25)
    seq_mask = torch.ones(B, L, dtype=torch.float32)
    extra_msa_mask = seq_mask[:, None, :].expand(B, E, L).clone()

    m_out, z_out = module(
        m,
        z,
        extra_msa_feat=extra_msa_feat,
        seq_mask=seq_mask,
        extra_msa_mask=extra_msa_mask,
    )

    assert m_out.shape == m.shape
    assert z_out.shape == z.shape
    assert torch.isfinite(m_out).all()
    assert torch.isfinite(z_out).all()
    assert not torch.allclose(z_out, z)


def test_alphafold2_optional_frontend_modules_smoke(toy_model, toy_batch):
    torch.manual_seed(8)

    toy_model.eval()
    batch_size, length = toy_batch["seq_tokens"].shape
    msa_depth = toy_batch["msa_tokens"].shape[1]
    num_templates = 2
    num_extra = 4

    extra_msa_feat = torch.randn(
        batch_size,
        num_extra,
        length,
        toy_model.extra_msa_stack.extra_proj.in_features,
    )
    extra_msa_mask = toy_batch["seq_mask"][:, None, :].expand(batch_size, num_extra, length).clone()
    template_angle_feat = torch.randn(
        batch_size,
        num_templates,
        length,
        toy_model.template_stack.angle_embedder.linear_1.in_features,
    )
    template_pair_feat = torch.randn(
        batch_size,
        num_templates,
        length,
        length,
        toy_model.template_stack.template_pair_proj.in_features,
    )
    template_mask = toy_batch["seq_mask"][:, None, :].expand(batch_size, num_templates, length).clone()

    with torch.no_grad():
        outputs = toy_model(
            seq_tokens=toy_batch["seq_tokens"],
            msa_tokens=toy_batch["msa_tokens"],
            seq_mask=toy_batch["seq_mask"],
            msa_mask=toy_batch["msa_mask"],
            ideal_backbone_local=toy_batch["ideal_backbone_local"],
            extra_msa_feat=extra_msa_feat,
            extra_msa_mask=extra_msa_mask,
            template_angle_feat=template_angle_feat,
            template_pair_feat=template_pair_feat,
            template_mask=template_mask,
        )

    assert outputs["m"].shape == (batch_size, msa_depth + num_templates, length, 256)
    assert outputs["z"].shape == (batch_size, length, length, 128)
    assert outputs["s"].shape == (batch_size, length, 256)
    assert outputs["backbone_coords"].shape == (batch_size, length, 4, 3)
    assert torch.isfinite(outputs["m"]).all()
    assert torch.isfinite(outputs["z"]).all()
    assert torch.isfinite(outputs["plddt"]).all()
