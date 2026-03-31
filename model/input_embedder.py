"""Input embedding layers for sequences and MSAs.

This module converts discrete residue tokens and MSA tokens into the initial
MSA and pair representations consumed by the Evoformer. It also handles masking
and relative positional features for the pair stream.
"""

import torch
import torch.nn as nn


class InputEmbedder(nn.Module):
    """
    Canonical AlphaFold2-style input embedder.

    Inputs:
        seq_tokens : [B, L]
        msa_tokens : [B, N_msa, L]
        seq_mask   : [B, L]          (optional)
        msa_mask   : [B, N_msa, L]   (optional)

    Outputs:
        m : [B, N_msa, L, c_m]       initial MSA representation
        z : [B, L, L, c_z]           initial pair representation
    """

    def __init__(
        self,
        n_tokens: int,
        c_m: int = 256,
        c_z: int = 128,
        c_s: int = 256,
        max_relpos: int = 32,
        pad_idx: int = 0):

        super().__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.c_s = c_s
        self.max_relpos = max_relpos

        # Token embeddings
        self.target_embedding = nn.Embedding(n_tokens, c_s, padding_idx=pad_idx)
        self.msa_embedding = nn.Embedding(n_tokens, c_m, padding_idx=pad_idx)

        # Project target sequence into MSA channel space
        self.target_to_msa = nn.Linear(c_s, c_m)

        # Pair initialization from target sequence
        self.left_single = nn.Linear(c_s, c_z)
        self.right_single = nn.Linear(c_s, c_z)

        # Relative positional encoding for pair representation
        self.relpos_embedding = nn.Embedding(2 * max_relpos + 1, c_z)

        self.m_ln = nn.LayerNorm(c_m)
        self.z_ln = nn.LayerNorm(c_z)

    def _make_relpos(self, L: int, device: torch.device):
        """
        Con esto sabemos que tan lejos esta el residuo j del i como pos embedding 
        asi el modelo aprende cuales estan cerca en secuencia.
        
        """
        idx = torch.arange(L, device=device)
        rel = idx[:, None] - idx[None, :]  # [L, L]
        rel = torch.clamp(rel, -self.max_relpos, self.max_relpos)
        rel = rel + self.max_relpos
        return rel

    def forward(
        self,
        seq_tokens: torch.Tensor,
        msa_tokens: torch.Tensor,
        seq_mask: torch.Tensor = None,
        msa_mask: torch.Tensor = None):

        B, N_msa, L = msa_tokens.shape
        device = msa_tokens.device

        # Target sequence embedding
        # s: [B, L, c_s]
        s = self.target_embedding(seq_tokens)

        # Initial MSA representation
        # m token embedding + target injection
        m = self.msa_embedding(msa_tokens)
        target_bias = self.target_to_msa(s)
        m = m + target_bias[:, None, :, :]

        m = self.m_ln(m)

        if msa_mask is not None:
            m = m * msa_mask.unsqueeze(-1)

        # Initial pair representation
        # z_ij = W_a s_i + W_b s_j + relpos(i-j)
        left = self.left_single(s)
        right = self.right_single(s)

        z = left[:, :, None, :] + right[:, None, :, :]

        relpos = self._make_relpos(L, device)
        relpos_emb = self.relpos_embedding(relpos)
        z = z + relpos_emb[None, :, :, :]

        z = self.z_ln(z)

        if seq_mask is not None:
            pair_mask = seq_mask[:, :, None] * seq_mask[:, None, :]  # [B, L, L]
            z = z * pair_mask.unsqueeze(-1)

        return m, z
