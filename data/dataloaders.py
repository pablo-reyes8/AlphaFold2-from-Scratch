from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from difflib import SequenceMatcher
from torch.utils.data import Dataset

from data.foldbench import build_manifest_dataframe, load_manifest_dataframe


AA_VOCAB = {
    "-": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "X": 21,
    "B": 22,
    "Z": 23,
    "U": 24,
    "O": 25,
    ".": 26,
}

UNK_TOKEN = AA_VOCAB["X"]


def tokenize_sequence(seq: str) -> torch.Tensor:
    return torch.tensor(
        [AA_VOCAB.get(character.upper(), UNK_TOKEN) for character in seq],
        dtype=torch.long,
    )


def read_a3m(a3m_path: str | Path, max_msa_seqs: int | None = None) -> list[str]:
    sequences: list[str] = []
    current_name: str | None = None
    current_sequence: list[str] = []

    with Path(a3m_path).expanduser().open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_name is not None:
                    sequence = "".join(current_sequence)
                    sequence = "".join(character for character in sequence if not character.islower())
                    sequences.append(sequence)
                    if max_msa_seqs is not None and len(sequences) >= max_msa_seqs:
                        break

                current_name = line[1:]
                current_sequence = []
                continue

            current_sequence.append(line)

    if (max_msa_seqs is None or len(sequences) < max_msa_seqs) and current_name is not None:
        sequence = "".join(current_sequence)
        sequence = "".join(character for character in sequence if not character.islower())
        sequences.append(sequence)

    return sequences


def pad_or_crop_msa(msa_seqs: list[str], target_len: int, max_msa_seqs: int) -> list[str]:
    fixed: list[str] = []

    for sequence in msa_seqs[:max_msa_seqs]:
        if len(sequence) < target_len:
            sequence = sequence + "-" * (target_len - len(sequence))
        elif len(sequence) > target_len:
            sequence = sequence[:target_len]
        fixed.append(sequence)

    if not fixed:
        fixed = ["-" * target_len]

    return fixed


def select_msa_sequences(
    msa_seqs: list[str],
    *,
    target_sequence: str,
    target_len: int,
    max_msa_seqs: int,
    single_sequence_mode: bool = False,
) -> list[str]:
    if single_sequence_mode:
        return pad_or_crop_msa([target_sequence], target_len=target_len, max_msa_seqs=1)

    return pad_or_crop_msa(msa_seqs, target_len=target_len, max_msa_seqs=max_msa_seqs)


def tokenize_msa(msa_seqs: list[str]) -> torch.Tensor:
    return torch.stack([tokenize_sequence(sequence) for sequence in msa_seqs], dim=0)


def _require_biopython():
    try:
        from Bio import Align
        from Bio.PDB.MMCIFParser import MMCIFParser
        from Bio.SeqUtils import seq1
    except ImportError as exc:
        raise ImportError(
            "Biopython is required for mmCIF parsing and sequence alignment. "
            "Install it with `pip install biopython`."
        ) from exc

    return Align, MMCIFParser, seq1


def safe_residue_to_aa(residue: Any, seq1_fn) -> str:
    resname = residue.get_resname().strip()
    try:
        aa = seq1_fn(resname)
        if len(aa) == 1:
            return aa
    except Exception:
        pass
    return "X"


@lru_cache(maxsize=128)
def _extract_chain_sequences_and_backbone_cached(cif_path: str) -> dict[str, dict[str, Any]]:
    _, mmcif_parser_cls, seq1_fn = _require_biopython()

    parser = mmcif_parser_cls(QUIET=True)
    structure = parser.get_structure(Path(cif_path).stem, cif_path)
    first_model = next(structure.get_models())

    out: dict[str, dict[str, Any]] = {}

    for chain in first_model:
        sequence_chars: list[str] = []
        coords_n: list[Any] = []
        coords_ca: list[Any] = []
        coords_c: list[Any] = []

        for residue in chain:
            if residue.id[0].strip() != "":
                continue

            sequence_chars.append(safe_residue_to_aa(residue, seq1_fn=seq1_fn))
            coords_n.append(residue["N"].coord if "N" in residue else [np.nan, np.nan, np.nan])
            coords_ca.append(residue["CA"].coord if "CA" in residue else [np.nan, np.nan, np.nan])
            coords_c.append(residue["C"].coord if "C" in residue else [np.nan, np.nan, np.nan])

        if not sequence_chars:
            continue

        out[chain.id] = {
            "sequence": "".join(sequence_chars),
            "coords_n": np.array(coords_n, dtype=np.float32),
            "coords_ca": np.array(coords_ca, dtype=np.float32),
            "coords_c": np.array(coords_c, dtype=np.float32),
        }

    return out


def extract_chain_sequences_and_backbone(cif_path: str | Path) -> dict[str, dict[str, Any]]:
    cached = _extract_chain_sequences_and_backbone_cached(str(Path(cif_path).expanduser()))
    copied: dict[str, dict[str, Any]] = {}

    for chain_id, info in cached.items():
        copied[chain_id] = {
            "sequence": info["sequence"],
            "coords_n": info["coords_n"].copy(),
            "coords_ca": info["coords_ca"].copy(),
            "coords_c": info["coords_c"].copy(),
        }

    return copied


def sequence_identity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_target_to_chain(
    target_seq: str,
    chain_data: dict[str, dict[str, Any]],
    min_identity: float = 0.85,
) -> tuple[str, float] | None:
    try:
        align_module, _, _ = _require_biopython()
        aligner = align_module.PairwiseAligner()
        aligner.mode = "local"
    except ImportError:
        aligner = None

    best_chain: str | None = None
    best_score = -1.0

    for chain_id, info in chain_data.items():
        chain_seq = info["sequence"]
        if not chain_seq:
            continue

        if aligner is not None:
            score = aligner.score(target_seq, chain_seq)
            normalized_score = score / max(1, len(chain_seq))
        else:
            normalized_score = sequence_identity(target_seq, chain_seq)

        if normalized_score > best_score:
            best_score = normalized_score
            best_chain = chain_id

    if best_chain is None or best_score < min_identity:
        return None

    return best_chain, float(best_score)


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    diffs = coords[:, None, :] - coords[None, :, :]
    return torch.sqrt(torch.sum(diffs**2, dim=-1) + 1e-8)


def dihedral_angle(p0, p1, p2, p3, eps: float = 1e-8):
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm < eps:
        return np.nan
    b1u = b1 / b1_norm

    v = b0 - np.dot(b0, b1u) * b1u
    w = b2 - np.dot(b2, b1u) * b1u

    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < eps or w_norm < eps:
        return np.nan

    x = np.dot(v, w)
    y = np.dot(np.cross(b1u, v), w)
    return np.arctan2(y, x)


def backbone_torsions_from_coords(
    coords_n,
    coords_ca,
    coords_c,
    valid_backbone_mask,
    eps: float = 1e-8,
):
    length = coords_n.shape[0]
    torsion_true = np.zeros((length, 3, 2), dtype=np.float32)
    torsion_mask = np.zeros((length, 3), dtype=np.float32)

    for index in range(length):
        if index > 0 and valid_backbone_mask[index - 1] and valid_backbone_mask[index]:
            angle = dihedral_angle(
                coords_c[index - 1],
                coords_n[index],
                coords_ca[index],
                coords_c[index],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 0, 0] = np.sin(angle)
                torsion_true[index, 0, 1] = np.cos(angle)
                torsion_mask[index, 0] = 1.0

        if index < length - 1 and valid_backbone_mask[index] and valid_backbone_mask[index + 1]:
            angle = dihedral_angle(
                coords_n[index],
                coords_ca[index],
                coords_c[index],
                coords_n[index + 1],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 1, 0] = np.sin(angle)
                torsion_true[index, 1, 1] = np.cos(angle)
                torsion_mask[index, 1] = 1.0

            angle = dihedral_angle(
                coords_ca[index],
                coords_c[index],
                coords_n[index + 1],
                coords_ca[index + 1],
                eps=eps,
            )
            if np.isfinite(angle):
                torsion_true[index, 2, 0] = np.sin(angle)
                torsion_true[index, 2, 1] = np.cos(angle)
                torsion_mask[index, 2] = 1.0

    return (
        torch.tensor(torsion_true, dtype=torch.float32),
        torch.tensor(torsion_mask, dtype=torch.float32),
    )


class FoldbenchProteinDataset(Dataset):
    def __init__(
        self,
        json_path: str | None = None,
        msa_root: str | None = None,
        cif_root: str | None = None,
        manifest_csv: str | None = None,
        max_msa_seqs: int = 128,
        use_a3m_name: str = "cfdb_hits.a3m",
        max_samples: int | None = None,
        min_identity: float = 0.90,
        single_sequence_mode: bool = False,
        verbose: bool = True,
    ):
        self.json_path = Path(json_path).expanduser() if json_path is not None else None
        self.msa_root = Path(msa_root).expanduser() if msa_root is not None else None
        self.cif_root = Path(cif_root).expanduser() if cif_root is not None else None
        self.manifest_csv = Path(manifest_csv).expanduser() if manifest_csv is not None else None
        self.max_msa_seqs = max_msa_seqs
        self.use_a3m_name = use_a3m_name
        self.min_identity = min_identity
        self.single_sequence_mode = bool(single_sequence_mode)

        self.manifest_df = self._load_manifest()
        rows, dropped = self._build_index(self.manifest_df)

        if max_samples is not None:
            rows = rows[:max_samples]

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        self.dropped = dropped

        if verbose:
            print(f"Dataset valid examples: {len(self.df)}")
            print(f"Dropped examples: {len(self.dropped)}")
            if not self.df.empty:
                print(
                    self.df[
                        ["query_name", "msa_chain_id", "matched_chain_id", "match_identity"]
                    ].head()
                )

    def _load_manifest(self) -> pd.DataFrame:
        if self.manifest_csv is not None:
            return load_manifest_dataframe(
                manifest_csv=self.manifest_csv,
                msa_root=self.msa_root,
                cif_root=self.cif_root,
            )

        if self.json_path is None or self.msa_root is None or self.cif_root is None:
            raise ValueError(
                "FoldbenchProteinDataset requires either manifest_csv or json_path + msa_root + cif_root."
            )

        return build_manifest_dataframe(
            json_path=self.json_path,
            msa_root=self.msa_root,
            cif_root=self.cif_root,
        )

    def _build_index(self, manifest_df: pd.DataFrame):
        rows: list[dict[str, Any]] = []
        dropped: list[tuple[str, str]] = []

        for row in manifest_df.to_dict(orient="records"):
            query_name = str(row.get("query_name", ""))
            target_sequence = str(row.get("sequence", "") or "")
            msa_chain_id = str(row.get("chain_id", "") or "")
            msa_dir = Path(str(row.get("msa_dir", ""))).expanduser()
            msa_file = msa_dir / self.use_a3m_name
            cif_value = row.get("cif_file")
            cif_file = Path(str(cif_value)).expanduser() if pd.notna(cif_value) and cif_value else None

            if not query_name:
                dropped.append((query_name, "missing_query_name"))
                continue
            if not target_sequence:
                dropped.append((query_name, "missing_target_sequence"))
                continue
            if cif_file is None or not cif_file.exists():
                dropped.append((query_name, "no_cif"))
                continue
            if not msa_file.exists():
                dropped.append((query_name, "no_msa"))
                continue

            try:
                chain_data = extract_chain_sequences_and_backbone(cif_file)
                match = match_target_to_chain(
                    target_seq=target_sequence,
                    chain_data=chain_data,
                    min_identity=self.min_identity,
                )
            except Exception as exc:
                dropped.append((query_name, f"parse_error:{exc}"))
                continue

            if match is None:
                dropped.append((query_name, "no_chain_match"))
                continue

            matched_chain_id, match_identity = match
            rows.append(
                {
                    "query_name": query_name,
                    "target_sequence": target_sequence,
                    "msa_chain_id": msa_chain_id,
                    "matched_chain_id": matched_chain_id,
                    "match_identity": match_identity,
                    "matched_chain_sequence": chain_data[matched_chain_id]["sequence"],
                    "msa_file": str(msa_file),
                    "cif_file": str(cif_file),
                }
            )

        return rows, dropped

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        query_name = row["query_name"]
        target_sequence = row["target_sequence"]
        matched_chain_id = row["matched_chain_id"]
        msa_file = Path(row["msa_file"])
        cif_file = Path(row["cif_file"])

        seq_tokens = tokenize_sequence(target_sequence)

        msa_seqs = read_a3m(msa_file, max_msa_seqs=self.max_msa_seqs)
        msa_seqs = select_msa_sequences(
            msa_seqs,
            target_sequence=target_sequence,
            target_len=len(target_sequence),
            max_msa_seqs=self.max_msa_seqs,
            single_sequence_mode=self.single_sequence_mode,
        )
        msa_tokens = tokenize_msa(msa_seqs)
        msa_mask = (msa_tokens != AA_VOCAB["-"]).float()

        chain_data = extract_chain_sequences_and_backbone(cif_file)
        chain_entry = chain_data[matched_chain_id]

        coords_n_np = chain_entry["coords_n"].copy()
        coords_ca_np = chain_entry["coords_ca"].copy()
        coords_c_np = chain_entry["coords_c"].copy()

        valid_n = ~np.isnan(coords_n_np).any(axis=1)
        valid_ca = ~np.isnan(coords_ca_np).any(axis=1)
        valid_c = ~np.isnan(coords_c_np).any(axis=1)

        valid_res_mask_np = valid_ca.astype(np.float32)
        valid_backbone_mask_np = (valid_n & valid_ca & valid_c).astype(np.float32)

        coords_n_np = np.nan_to_num(coords_n_np, nan=0.0)
        coords_ca_np = np.nan_to_num(coords_ca_np, nan=0.0)
        coords_c_np = np.nan_to_num(coords_c_np, nan=0.0)

        length = min(
            len(seq_tokens),
            coords_ca_np.shape[0],
            coords_n_np.shape[0],
            coords_c_np.shape[0],
            msa_tokens.shape[1],
        )

        seq_tokens = seq_tokens[:length]
        msa_tokens = msa_tokens[:, :length]
        msa_mask = msa_mask[:, :length]

        coords_n_np = coords_n_np[:length]
        coords_ca_np = coords_ca_np[:length]
        coords_c_np = coords_c_np[:length]

        valid_res_mask_np = valid_res_mask_np[:length]
        valid_backbone_mask_np = valid_backbone_mask_np[:length]

        torsion_true, torsion_mask = backbone_torsions_from_coords(
            coords_n=coords_n_np,
            coords_ca=coords_ca_np,
            coords_c=coords_c_np,
            valid_backbone_mask=valid_backbone_mask_np.astype(bool),
        )

        coords_n = torch.tensor(coords_n_np, dtype=torch.float32)
        coords_ca = torch.tensor(coords_ca_np, dtype=torch.float32)
        coords_c = torch.tensor(coords_c_np, dtype=torch.float32)

        valid_res_mask = torch.tensor(valid_res_mask_np, dtype=torch.float32)
        valid_backbone_mask = torch.tensor(valid_backbone_mask_np, dtype=torch.float32)
        dist_map = pairwise_distances(coords_ca)

        return {
            "id": query_name,
            "msa_chain_id": row["msa_chain_id"],
            "matched_chain_id": matched_chain_id,
            "match_identity": torch.tensor(row["match_identity"], dtype=torch.float32),
            "sequence_str": target_sequence[:length],
            "seq_tokens": seq_tokens,
            "msa_tokens": msa_tokens,
            "msa_mask": msa_mask,
            "coords_n": coords_n,
            "coords_ca": coords_ca,
            "coords_c": coords_c,
            "dist_map": dist_map,
            "valid_res_mask": valid_res_mask,
            "valid_backbone_mask": valid_backbone_mask,
            "torsion_true": torsion_true,
            "torsion_mask": torsion_mask,
        }
