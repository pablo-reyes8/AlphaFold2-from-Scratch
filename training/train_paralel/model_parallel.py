"""Two-stage model-parallel wrappers for the AlphaFold2 top-level module.

This module keeps Evoformer and recycling updates on a first device, moves the
structure module plus heads to a second device, and preserves the original
state-dict surface so checkpoints remain compatible with the plain model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.ipa_transformations import apply_transform


class AlphaFold2ModelParallel(nn.Module):
    """Wrap an existing ``AlphaFold2`` instance into a simple two-stage pipeline."""

    def __init__(self, model: nn.Module, stage_devices: tuple[str | torch.device, ...]):
        super().__init__()
        if len(stage_devices) == 0:
            raise ValueError("stage_devices must contain at least one device.")

        devices = tuple(torch.device(device) for device in stage_devices)
        self.stage_devices = devices
        self.input_device = devices[0]
        self.output_device = devices[-1]

        self.c_z = model.c_z
        self.recycle_min_bin = float(model.recycle_min_bin)
        self.recycle_max_bin = float(model.recycle_max_bin)
        self.recycle_dist_bins = int(model.recycle_dist_bins)
        self.evoformer_enabled = bool(getattr(model, "evoformer_enabled", True))
        self.recycle_pair_enabled = bool(getattr(model, "recycle_pair_enabled", True))
        self.recycle_position_enabled = bool(getattr(model, "recycle_position_enabled", True))
        self.structure_pair_context_enabled = bool(getattr(model, "structure_pair_context_enabled", True))
        self.distogram_head_enabled = bool(getattr(model, "distogram_head_enabled", True))
        self.plddt_head_enabled = bool(getattr(model, "plddt_head_enabled", True))
        self.torsion_head_enabled = bool(getattr(model, "torsion_head_enabled", True))

        self.input_embedder = model.input_embedder.to(self.input_device)
        self.evoformer = model.evoformer.to(self.input_device)
        self.recycle_pair_norm = model.recycle_pair_norm.to(self.input_device)
        self.recycle_pos_embedding = model.recycle_pos_embedding.to(self.input_device)

        self.single_proj = model.single_proj.to(self.output_device)
        self.structure_module = model.structure_module.to(self.output_device)
        self.plddt_head = model.plddt_head.to(self.output_device)
        self.distogram_head = model.distogram_head.to(self.output_device)
        self.torsion_head = model.torsion_head.to(self.output_device)

    def _to_input_device(self, tensor):
        if tensor is None:
            return None
        return tensor.to(self.input_device, non_blocking=True)

    def _to_output_device(self, tensor):
        if tensor is None:
            return None
        return tensor.to(self.output_device, non_blocking=True)

    def _apply_recycle_pair_update(self, z, prev_pair, pair_mask=None):
        if (not self.recycle_pair_enabled) or (prev_pair is None):
            return z

        z = z + self.recycle_pair_norm(prev_pair)

        if pair_mask is not None:
            z = z * pair_mask.unsqueeze(-1)

        return z

    def _positions_to_recycle_dgram(self, positions, dtype, pair_mask=None):
        deltas = positions[:, :, None, :] - positions[:, None, :, :]
        sq_dist = deltas.pow(2).sum(dim=-1).float()

        boundaries = torch.linspace(
            self.recycle_min_bin,
            self.recycle_max_bin,
            self.recycle_dist_bins - 1,
            device=positions.device,
            dtype=sq_dist.dtype,
        ).pow(2)

        bin_ids = torch.bucketize(sq_dist, boundaries)
        recycle_update = self.recycle_pos_embedding(bin_ids).to(dtype=dtype)

        if pair_mask is not None:
            recycle_update = recycle_update * pair_mask.unsqueeze(-1)

        return recycle_update

    def _extract_recycle_positions(self, backbone_coords, t):
        if backbone_coords is not None:
            ca_index = 1 if backbone_coords.shape[-2] > 1 else 0
            return backbone_coords[:, :, ca_index, :]
        return t

    def _build_structure_pair_input(self, z):
        if self.structure_pair_context_enabled:
            return z
        return torch.zeros_like(z)

    def forward(
        self,
        seq_tokens,
        msa_tokens,
        seq_mask=None,
        msa_mask=None,
        ideal_backbone_local=None,
        num_recycles: int = 0,
    ):
        seq_tokens = self._to_input_device(seq_tokens)
        msa_tokens = self._to_input_device(msa_tokens)
        seq_mask_input = self._to_input_device(seq_mask)
        msa_mask_input = self._to_input_device(msa_mask)

        if seq_mask_input is not None:
            pair_mask_input = seq_mask_input[:, :, None] * seq_mask_input[:, None, :]
        else:
            pair_mask_input = None

        num_recycles = max(0, int(num_recycles))
        prev_pair = None
        prev_positions = None
        outputs = None

        for recycle_idx in range(num_recycles + 1):
            m, z = self.input_embedder(
                seq_tokens=seq_tokens,
                msa_tokens=msa_tokens,
                seq_mask=seq_mask_input,
                msa_mask=msa_mask_input,
            )

            z = self._apply_recycle_pair_update(
                z,
                prev_pair=prev_pair,
                pair_mask=pair_mask_input,
            )

            if self.recycle_position_enabled and prev_positions is not None:
                z = z + self._positions_to_recycle_dgram(
                    prev_positions,
                    dtype=z.dtype,
                    pair_mask=pair_mask_input,
                )

            if self.evoformer_enabled:
                m, z = self.evoformer(
                    m,
                    z,
                    msa_mask=msa_mask_input,
                    pair_mask=pair_mask_input,
                )

            m_output = self._to_output_device(m)
            z_output = self._to_output_device(z)
            seq_mask_output = self._to_output_device(seq_mask_input)

            distogram_logits = self.distogram_head(z_output) if self.distogram_head_enabled else None
            s0 = self.single_proj(m_output)
            structure_pair = self._build_structure_pair_input(z_output)
            s, R, t = self.structure_module(s0, structure_pair, mask=seq_mask_output)

            backbone_coords = None
            if ideal_backbone_local is not None:
                ideal_backbone_output = self._to_output_device(ideal_backbone_local)
                if ideal_backbone_output.dim() == 2:
                    ideal_backbone_output = ideal_backbone_output.unsqueeze(0).unsqueeze(0)
                elif ideal_backbone_output.dim() != 4:
                    raise ValueError("ideal_backbone_local must have shape [A,3] or [B,L,A,3]")

                if ideal_backbone_output.shape[0] == 1 and ideal_backbone_output.shape[1] == 1:
                    batch_size, length = seq_tokens.shape
                    ideal_backbone_output = ideal_backbone_output.expand(batch_size, length, -1, -1)

                backbone_coords = apply_transform(
                    R[:, :, None, :, :],
                    t[:, :, None, :],
                    ideal_backbone_output,
                )

            torsions = self.torsion_head(s0, s, mask=seq_mask_output) if self.torsion_head_enabled else None
            if self.plddt_head_enabled:
                plddt_logits, plddt = self.plddt_head(s)
            else:
                plddt_logits, plddt = None, None

            outputs = {
                "m": m_output,
                "z": z_output,
                "s": s,
                "R": R,
                "t": t,
                "backbone_coords": backbone_coords,
                "torsions": torsions,
                "plddt_logits": plddt_logits,
                "plddt": plddt,
                "distogram_logits": distogram_logits,
            }

            if recycle_idx < num_recycles:
                prev_pair = z.detach()
                prev_positions = self._extract_recycle_positions(backbone_coords, t).detach()
                prev_positions = self._to_input_device(prev_positions)

        return outputs


def build_model_parallel_wrapper(
    model: nn.Module,
    stage_devices: tuple[str | torch.device, ...],
) -> AlphaFold2ModelParallel:
    """Create a two-stage model-parallel wrapper around an existing model."""
    return AlphaFold2ModelParallel(model=model, stage_devices=stage_devices)
