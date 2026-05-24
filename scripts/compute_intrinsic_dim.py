"""
Compute layer-wise intrinsic dimensionality from pre-extracted embeddings.

Example:
    python scripts/compute_intrinsic_dim.py \
        --embeddings data/experiment3/M02_qwen3-4b-embedding_passages.npz \
        --output-dir results/intrinsic_dim/experiment3/M02_qwen3-4b-embedding_passages
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embedding import load_embeddings
from src.intrinsic_dim import compute_id_per_layer, save_id_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings}")
    layer_embeddings, metadata = load_embeddings(args.embeddings)
    n_layers, n_samples, hidden_dim = layer_embeddings.shape
    print(f"  shape: {layer_embeddings.shape} "
          f"({n_layers} layers, {n_samples} stimuli, {hidden_dim} dims)")

    print("Computing I_d per layer...")
    id_values = compute_id_per_layer(layer_embeddings)

    save_id_results(
        output_dir / 'intrinsic_dim.npz',
        id_values,
        metadata=dict(metadata) if metadata else None,
    )

    # readable summary
    summary_path = output_dir / 'intrinsic_dim_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("layer\tintrinsic_dim\n")
        for i, val in enumerate(id_values):
            f.write(f"{i}\t{val:.4f}\n")

    print(f"\nSaved to {output_dir / 'intrinsic_dim.npz'}")
    print(f"Summary: {summary_path}\n")
    print("Layer   I_d")
    for i, val in enumerate(id_values):
        print(f"{i:5d}  {val:7.2f}")

    peak = int(np.argmax(id_values))
    print(f"\nPeak I_d: layer {peak} ({id_values[peak]:.2f})")


if __name__ == '__main__':
    main()
