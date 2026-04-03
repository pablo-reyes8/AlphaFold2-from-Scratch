"""Top-level AlphaFold2-like model assembly.

This module wires the input embedder, Evoformer stack, structure module, and
output heads into a single PyTorch model that returns representations,
geometric predictions, backbone coordinates, torsions, confidence, and
distogram outputs.
"""

import torch 
import torch.nn as nn 
from model.evoformer_block import * 
from model.evoformer_stack import *

from model.alphafold2_heads import * 
from model.torsion_head import *
from model.structure_block import *

class AlphaFold2(nn.Module):
    """
    AF2-like model.

    Outputs:
      - m, z
      - single representation s
      - frames R, t
      - backbone coords from ideal local coords
      - torsion angles
      - pLDDT
      - distogram logits
    """
    @staticmethod
    def _normalize_ablation_id(ablation):
        if ablation is None:
            return None

        if isinstance(ablation, str):
            digits = "".join(character for character in ablation if character.isdigit())
            if digits == "":
                raise ValueError(f"Unsupported ablation identifier: {ablation}")
            return int(digits)

        return int(ablation)

    @classmethod
    def resolve_ablation_defaults(cls, ablation):
        ablation_id = cls._normalize_ablation_id(ablation)
        mapping = {
            None: {},
            1: {
                "evoformer_pair_stack_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
                "plddt_head_enabled": False,
            },
            2: {
                "evoformer_triangle_attention_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
                "plddt_head_enabled": False,
            },
            3: {
                "distogram_head_enabled": False,
                "plddt_head_enabled": False,
                "torsion_head_enabled": False,
            },
            4: {
                "use_block_specific_params": True,
            },
            5: {
                "evoformer_enabled": False,
                "recycle_pair_enabled": False,
                "recycle_position_enabled": False,
            },
        }
        if ablation_id not in mapping:
            valid = ", ".join(str(key) for key in sorted(key for key in mapping if key is not None))
            raise ValueError(f"Unsupported AlphaFold2 ablation '{ablation}'. Valid ids: {valid}")
        return mapping[ablation_id]

    def __init__(
        self,
        n_tokens,
        c_m=256,
        c_z=128,
        c_s=256,
        max_relpos=32,
        pad_idx=0,
        num_evoformer_blocks=4,
        num_structure_blocks=8,
        transition_expansion_evoformer = 4, 
        transition_expansion_structure = 4, 
        use_block_specific_params = False, 
        dist_bins=64,
        plddt_bins=50,
        n_torsions=7,
        num_res_blocks_torsion=2,
        recycle_min_bin=3.25,
        recycle_max_bin=20.75,
        recycle_dist_bins=15,
        ablation=None,
        evoformer_enabled=True,
        evoformer_pair_stack_enabled=True,
        evoformer_triangle_multiplication_enabled=True,
        evoformer_triangle_attention_enabled=True,
        evoformer_pair_transition_enabled=True,
        recycle_pair_enabled=True,
        recycle_position_enabled=True,
        structure_pair_context_enabled=True,
        distogram_head_enabled=True,
        plddt_head_enabled=True,
        torsion_head_enabled=True):

        super().__init__()
        ablation_defaults = self.resolve_ablation_defaults(ablation)
        use_block_specific_params = ablation_defaults.get("use_block_specific_params", use_block_specific_params)
        evoformer_enabled = ablation_defaults.get("evoformer_enabled", evoformer_enabled)
        evoformer_pair_stack_enabled = ablation_defaults.get(
            "evoformer_pair_stack_enabled",
            evoformer_pair_stack_enabled,
        )
        evoformer_triangle_multiplication_enabled = ablation_defaults.get(
            "evoformer_triangle_multiplication_enabled",
            evoformer_triangle_multiplication_enabled,
        )
        evoformer_triangle_attention_enabled = ablation_defaults.get(
            "evoformer_triangle_attention_enabled",
            evoformer_triangle_attention_enabled,
        )
        evoformer_pair_transition_enabled = ablation_defaults.get(
            "evoformer_pair_transition_enabled",
            evoformer_pair_transition_enabled,
        )
        recycle_pair_enabled = ablation_defaults.get("recycle_pair_enabled", recycle_pair_enabled)
        recycle_position_enabled = ablation_defaults.get("recycle_position_enabled", recycle_position_enabled)
        structure_pair_context_enabled = ablation_defaults.get(
            "structure_pair_context_enabled",
            structure_pair_context_enabled,
        )
        distogram_head_enabled = ablation_defaults.get("distogram_head_enabled", distogram_head_enabled)
        plddt_head_enabled = ablation_defaults.get("plddt_head_enabled", plddt_head_enabled)
        torsion_head_enabled = ablation_defaults.get("torsion_head_enabled", torsion_head_enabled)

        self.ablation = self._normalize_ablation_id(ablation)
        self.c_z = c_z
        self.recycle_min_bin = float(recycle_min_bin)
        self.recycle_max_bin = float(recycle_max_bin)
        self.recycle_dist_bins = int(recycle_dist_bins)
        self.evoformer_enabled = bool(evoformer_enabled)
        self.evoformer_pair_stack_enabled = bool(evoformer_pair_stack_enabled)
        self.evoformer_triangle_multiplication_enabled = bool(evoformer_triangle_multiplication_enabled)
        self.evoformer_triangle_attention_enabled = bool(evoformer_triangle_attention_enabled)
        self.evoformer_pair_transition_enabled = bool(evoformer_pair_transition_enabled)
        self.recycle_pair_enabled = bool(recycle_pair_enabled)
        self.recycle_position_enabled = bool(recycle_position_enabled)
        self.structure_pair_context_enabled = bool(structure_pair_context_enabled)
        self.distogram_head_enabled = bool(distogram_head_enabled)
        self.plddt_head_enabled = bool(plddt_head_enabled)
        self.torsion_head_enabled = bool(torsion_head_enabled)


        # Tokens de Entrada
        self.input_embedder = InputEmbedder(
            n_tokens=n_tokens,
            c_m=c_m,
            c_z=c_z,
            c_s=c_s,
            max_relpos=max_relpos,
            pad_idx=pad_idx)


        # Evoformer para m y z
        self.evoformer = EvoformerStack(
            num_blocks=num_evoformer_blocks,
            c_m=c_m,
            c_z=c_z,
            transition_expansion=transition_expansion_evoformer,
            pair_stack_enabled=self.evoformer_pair_stack_enabled,
            triangle_multiplication_enabled=self.evoformer_triangle_multiplication_enabled,
            triangle_attention_enabled=self.evoformer_triangle_attention_enabled,
            pair_transition_enabled=self.evoformer_pair_transition_enabled)

        self.single_proj = SingleProjection(c_m=c_m, c_s=c_s)

        # Structure Model con IPA
        self.structure_module = StructureModule(
            c_s=c_s,
            c_z=c_z,
            num_blocks=num_structure_blocks , transition_expansion=transition_expansion_structure , use_block_specific_params=use_block_specific_params)

        # Cabezas finales para entender el modelo
        self.plddt_head = PlddtHead(c_s=c_s, num_bins=plddt_bins)
        self.distogram_head = DistogramHead(c_z=c_z, num_bins=dist_bins)
        self.torsion_head = TorsionHead(c_s=c_s, n_torsions=n_torsions , num_res_blocks = num_res_blocks_torsion)
        self.recycle_pair_norm = nn.LayerNorm(c_z)
        self.recycle_pos_embedding = nn.Embedding(self.recycle_dist_bins, c_z)

        self._freeze_module(self.evoformer, enabled=self.evoformer_enabled)
        self._freeze_module(self.recycle_pair_norm, enabled=self.recycle_pair_enabled)
        self._freeze_module(self.recycle_pos_embedding, enabled=self.recycle_position_enabled)
        self._freeze_module(self.distogram_head, enabled=self.distogram_head_enabled)
        self._freeze_module(self.plddt_head, enabled=self.plddt_head_enabled)
        self._freeze_module(self.torsion_head, enabled=self.torsion_head_enabled)

    @staticmethod
    def _freeze_module(module, *, enabled):
        if enabled:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False

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
        num_recycles: int = 0):
        """
        ideal_backbone_local: [A, 3] or [1,1,A,3] or [B,L,A,3]
          e.g. local ideal coordinates for backbone atoms (N, CA, C, O)

        num_recycles:
          Number of extra recycling passes on the same batch. ``0`` means a
          single forward pass with no recycling.

        returns dict
        """
        if seq_mask is not None:
            pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]
        else:
            pair_mask = None

        num_recycles = max(0, int(num_recycles))
        prev_pair = None
        prev_positions = None
        outputs = None

        for recycle_idx in range(num_recycles + 1):
            # input
            m, z = self.input_embedder(
                seq_tokens=seq_tokens,
                msa_tokens=msa_tokens,
                seq_mask=seq_mask,
                msa_mask=msa_mask)

            z = self._apply_recycle_pair_update(
                z,
                prev_pair=prev_pair,
                pair_mask=pair_mask,
            )

            if self.recycle_position_enabled and prev_positions is not None:
                z = z + self._positions_to_recycle_dgram(
                    prev_positions,
                    dtype=z.dtype,
                    pair_mask=pair_mask,
                )

            # evoformer
            if self.evoformer_enabled:
                m, z = self.evoformer(
                    m,
                    z,
                    msa_mask=msa_mask,
                    pair_mask=pair_mask,)

            # z before structure for distogram
            distogram_logits = self.distogram_head(z) if self.distogram_head_enabled else None

            # single repr + structure
            s0 = self.single_proj(m)
            structure_pair = self._build_structure_pair_input(z)
            s, R, t = self.structure_module(s0, structure_pair, mask=seq_mask)

            # backbone coordinates from ideal local atoms
            backbone_coords = None
            if ideal_backbone_local is not None:
                if ideal_backbone_local.dim() == 2:
                    # [A,3] -> [1,1,A,3]
                    ideal_backbone_local = ideal_backbone_local.unsqueeze(0).unsqueeze(0)
                elif ideal_backbone_local.dim() == 4:
                    pass
                else:
                    raise ValueError("ideal_backbone_local must have shape [A,3] or [B,L,A,3]")

                if ideal_backbone_local.shape[0] == 1 and ideal_backbone_local.shape[1] == 1:
                    B, L = seq_tokens.shape
                    ideal_backbone_local = ideal_backbone_local.expand(B, L, -1, -1)

                backbone_coords = apply_transform(
                    R[:, :, None, :, :],     # [B,L,1,3,3]
                    t[:, :, None, :],        # [B,L,1,3]
                    ideal_backbone_local     # [B,L,A,3]
                )

            # torsions and confidence
            s_initial = s0
            s_final = s
            torsions = self.torsion_head(s_initial, s_final, mask=seq_mask) if self.torsion_head_enabled else None
            if self.plddt_head_enabled:
                plddt_logits, plddt = self.plddt_head(s)
            else:
                plddt_logits, plddt = None, None

            outputs = {
                "m": m,
                "z": z,
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

        return outputs
