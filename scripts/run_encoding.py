"""
Fit ridge regression encoding models, one per layer, for a participant.

Loads pre-computed embeddings from extract_embeddings.py and saves per-layer
accuracy arrays plus a summary (mean test accuracy per layer).

Example:
    python scripts/run_encoding.py \
        --mat-path M02/data_243sentences.mat \
        --embeddings data/M02_qwen3_passages.npz \
        --region languageLH \
        --output-dir results/M02_qwen3_passages

    To also save per-voxel accuracies use flag:
        --save-voxelwise

    With --save-voxelwise the output directory will contain:
        layer_summary.txt               – mean train/test accuracy per layer
        layer_accuracies.npz            – mean accuracies (always written)
        layer_accuracies_extended.npz   – full (n_layers, n_splits, n_voxels)
                                          arrays (only with --save-voxelwise)

    Re-running with --save-voxelwise will add layer_accuracies_extended.npz
    without overwriting the other two files.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import AVAILABLE_REGIONS, load_stimuli_and_responses
from src.embedding import load_embeddings
from src.encoding import fit_encoding_per_layer, summarize_layer_accuracies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-path', required=True)
    parser.add_argument('--embeddings', required=True,
                        help="Path to the .npz produced by extract_embeddings.py.")
    parser.add_argument('--level', choices=['passage', 'sentence'],
                        default='passage')
    parser.add_argument('--region', choices=AVAILABLE_REGIONS,
                        default='languageLH',
                        help="Atlas region to fit against. Must match the "
                             "region used when extracting the embeddings' "
                             "paired brain responses.")
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--save-voxelwise', action='store_true',
                        help="If set, save the full per-voxel, per-fold "
                             "accuracy arrays (n_layers, n_splits, n_voxels) "
                             "as layer_accuracies_extended.npz inside "
                             "--output-dir, alongside the standard outputs. "
                             "The standard layer_accuracies.npz and "
                             "layer_summary.txt are always written regardless.")
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading brain responses from {args.mat_path} "
          f"(region={args.region})")
    _, brain_responses = load_stimuli_and_responses(
        args.mat_path, level=args.level, region=args.region
    )
    print(f"  brain responses shape: {brain_responses.shape}")

    print(f"Loading embeddings from {args.embeddings}")
    layer_embeddings, metadata = load_embeddings(args.embeddings)
    print(f"  embeddings shape: {layer_embeddings.shape}")

    meta_region = metadata.get('region') if isinstance(metadata, dict) else None
    if meta_region and meta_region != args.region:
        print(f"  WARNING: embeddings were extracted with region="
              f"{meta_region!r} but --region={args.region!r}. "
              f"This is only OK if the embeddings themselves are "
              f"region-independent (they should be).")

    if layer_embeddings.shape[1] != brain_responses.shape[0]:
        raise ValueError(
            f"Stimulus count mismatch: embeddings have "
            f"{layer_embeddings.shape[1]} stimuli but brain responses have "
            f"{brain_responses.shape[0]}. Check --level matches the embeddings."
        )

    print(f"Fitting encoding models for {layer_embeddings.shape[0]} layers...")
    result = fit_encoding_per_layer(
        layer_embeddings,
        brain_responses,
        n_splits=args.n_splits,
        verbose=not args.quiet,
    )

    accs_train = result['accs_train']
    accs_test = result['accs_test']

    layer_means_test = summarize_layer_accuracies(accs_test)
    layer_means_train = summarize_layer_accuracies(accs_train)

    summary_path = output_dir / 'layer_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"# region: {args.region}\n")
        f.write(f"# n_voxels: {brain_responses.shape[1]}\n")
        f.write("layer\ttrain_acc\ttest_acc\n")
        for i, (tr, te) in enumerate(zip(layer_means_train, layer_means_test)):
            f.write(f"{i}\t{tr:.4f}\t{te:.4f}\n")
    print(f"Saved summary        -> {summary_path}")

    accs_path = output_dir / 'layer_accuracies.npz'
    np.savez_compressed(
        accs_path,
        layer_means_train=layer_means_train,
        layer_means_test=layer_means_test,
        region=args.region,
        n_voxels=brain_responses.shape[1],
    )
    print(f"Saved layer means    -> {accs_path}")


    if args.save_voxelwise:
        extended_path = output_dir / 'layer_accuracies_extended.npz'
        np.savez_compressed(
            extended_path,
            accs_train=accs_train,
            accs_test=accs_test,
            region=args.region,
            n_voxels=brain_responses.shape[1],
        )
        print(f"Saved voxelwise full -> {extended_path}")

    print()
    print("Layer   Train    Test")
    for i, (tr, te) in enumerate(zip(layer_means_train, layer_means_test)):
        print(f"{i:5d}  {tr:6.3f}  {te:6.3f}")

    best_layer = int(np.argmax(layer_means_test))
    print()
    print(f"Best layer (by mean test accuracy): {best_layer} "
          f"({layer_means_test[best_layer]:.3f})")


if __name__ == '__main__':
    main()