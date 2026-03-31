"""CLI for lightweight dataset inspection and 3D structure previews.

This script exposes concise dataset summaries, MSA previews, distance-map
rendering, dataloader smoke checks, and a simple 3D backbone visualization for
quick sanity checks before training.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.dataloaders import extract_chain_sequences_and_backbone
from data.visualize_data import manifest_summary, msa_preview, save_distance_map_figure
from scripts.common import build_dataloader_from_config, build_dataset_from_config, load_yaml_config, summarize_batch, summarize_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Foldbench data artifacts and dataloaders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summary_cmd = subparsers.add_parser("manifest-summary", help="Print a summary of a CSV manifest.")
    summary_cmd.add_argument("--manifest-csv", required=True, type=str)

    msa_cmd = subparsers.add_parser("msa-preview", help="Preview sequences stored in an A3M file.")
    msa_cmd.add_argument("--a3m-path", required=True, type=str)
    msa_cmd.add_argument("--limit", type=int, default=5)

    loader_cmd = subparsers.add_parser("loader-preview", help="Print dataset and batch summaries from the config.")
    loader_cmd.add_argument("--config", type=str, default="config/experiments/af2_poc.yaml")
    loader_cmd.add_argument("--manifest-csv", type=str, default=None)
    loader_cmd.add_argument("--batch-size", type=int, default=None)
    loader_cmd.add_argument("--max-samples", type=int, default=2)

    dmap_cmd = subparsers.add_parser("distance-map", help="Render a CA distance map from an mmCIF file.")
    dmap_cmd.add_argument("--cif-path", required=True, type=str)
    dmap_cmd.add_argument("--chain-id", required=True, type=str)
    dmap_cmd.add_argument("--output", required=True, type=str)

    backbone_cmd = subparsers.add_parser("protein-3d", help="Render a simple 3D backbone trace from an mmCIF file.")
    backbone_cmd.add_argument("--cif-path", required=True, type=str)
    backbone_cmd.add_argument("--chain-id", required=True, type=str)
    backbone_cmd.add_argument("--output", required=True, type=str)

    return parser.parse_args(argv)


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for rendering figures. Install the repo requirements first."
        ) from exc

    return plt


def save_backbone_3d_figure(
    cif_path: str | Path,
    chain_id: str,
    output_path: str | Path,
) -> Path:
    plt = _require_matplotlib()

    chain_data = extract_chain_sequences_and_backbone(cif_path)
    if chain_id not in chain_data:
        raise KeyError(f"Chain '{chain_id}' not found in {cif_path}")

    coords_n = chain_data[chain_id]["coords_n"]
    coords_ca = chain_data[chain_id]["coords_ca"]
    coords_c = chain_data[chain_id]["coords_c"]

    valid_ca = ~np.isnan(coords_ca).any(axis=1)
    valid_n = ~np.isnan(coords_n).any(axis=1)
    valid_c = ~np.isnan(coords_c).any(axis=1)

    coords_ca = coords_ca[valid_ca]
    coords_n = coords_n[valid_n]
    coords_c = coords_c[valid_c]

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if len(coords_ca) > 0:
        ax.plot(coords_ca[:, 0], coords_ca[:, 1], coords_ca[:, 2], color="royalblue", linewidth=2.0, label="CA trace")
        ax.scatter(coords_ca[0, 0], coords_ca[0, 1], coords_ca[0, 2], color="green", s=40, label="start")
        ax.scatter(coords_ca[-1, 0], coords_ca[-1, 1], coords_ca[-1, 2], color="red", s=40, label="end")

    if len(coords_n) > 0:
        ax.scatter(coords_n[:, 0], coords_n[:, 1], coords_n[:, 2], color="orange", s=8, alpha=0.35, label="N")
    if len(coords_c) > 0:
        ax.scatter(coords_c[:, 0], coords_c[:, 1], coords_c[:, 2], color="gray", s=8, alpha=0.35, label="C")

    ax.set_title(f"Backbone 3D preview - chain {chain_id}")
    ax.set_xlabel("x (A)")
    ax.set_ylabel("y (A)")
    ax.set_zlabel("z (A)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    return output


def run_loader_preview(args: argparse.Namespace) -> None:
    config = load_yaml_config(args.config)
    dataset = build_dataset_from_config(
        config,
        manifest_csv=args.manifest_csv,
        max_samples=args.max_samples,
        verbose=False,
    )
    loader = build_dataloader_from_config(
        dataset,
        config,
        batch_size=args.batch_size,
        shuffle=False,
    )
    print(f"[scripts.inspect_data] dataset summary: {summarize_dataset(dataset)}")
    if len(dataset) == 0:
        raise ValueError("Loader preview found zero valid examples. Check the manifest paths and drop reasons above.")
    print(f"[scripts.inspect_data] batch summary: {summarize_batch(next(iter(loader)))}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "manifest-summary":
        print(manifest_summary(args.manifest_csv))
        return

    if args.command == "msa-preview":
        print(msa_preview(args.a3m_path, limit=args.limit))
        return

    if args.command == "loader-preview":
        run_loader_preview(args)
        return

    if args.command == "distance-map":
        output = save_distance_map_figure(args.cif_path, args.chain_id, args.output)
        print(f"[scripts.inspect_data] saved distance map to {output}")
        return

    if args.command == "protein-3d":
        output = save_backbone_3d_figure(args.cif_path, args.chain_id, args.output)
        print(f"[scripts.inspect_data] saved 3D backbone figure to {output}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
