"""
Extract layer-wise embeddings for a participant's stimuli.

Saves a .npz file with an array of shape (n_layers, n_stimuli, hidden_dim).

Example:
    python scripts/extract_embeddings.py \
        --mat-path M02/data_243sentences.mat \
        --model-name Qwen/Qwen3-Embedding-8B \
        --pooling last \
        --region languageLH \
        --output data/M02_qwen3_passages.npz
"""

import argparse
import sys
from pathlib import Path

import torch

# Make src importable when running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loading import AVAILABLE_REGIONS, load_stimuli_and_responses
from src.embedding import encode_all_layers, load_model, save_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat-path', required=True,
                        help="Path to a participant's .mat file.")
    parser.add_argument('--model-name', required=True,
                        help="HuggingFace model identifier.")
    parser.add_argument('--level', choices=['passage', 'sentence'],
                        default='passage')
    parser.add_argument('--region', choices=AVAILABLE_REGIONS,
                        default='languageLH',
                        help="Atlas region to restrict voxels to. "
                             "Must match a name in data['meta']['atlases'].")
    parser.add_argument('--pooling', choices=['last', 'cls', 'mean'],
                        default='last',
                        help="Sequence pooling. 'last' for decoder-only "
                             "embedding models (e.g. Qwen3-Embedding), "
                             "'cls' for BERT-style, 'mean' is often used "
                             "in the brain encoding literature.")
    parser.add_argument('--scramble', action='store_true',
                        help="Bag-of-words ablation: shuffle the whitespace "
                             "tokens within each stimulus before encoding. "
                             "Punctuation stays attached to its token.")
    parser.add_argument('--scramble-seed', type=int, default=42,
                        help="Base seed for reproducible scrambling.")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max-length', type=int, default=None)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'],
                        default='float16',
                        help="Dtype to load the model in. float16 is needed "
                             "for large (>7B) models.")
    parser.add_argument('--output', required=True,
                        help="Output .npz path.")
    return parser.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading stimuli and brain responses from {args.mat_path} "
          f"(region={args.region}, scramble={args.scramble})")
    texts, brain_responses = load_stimuli_and_responses(
        args.mat_path,
        level=args.level,
        region=args.region,
        scramble=args.scramble,
        scramble_seed=args.scramble_seed,
    )
    print(f"  {len(texts)} {args.level}s, brain responses shape "
          f"{brain_responses.shape}")
    if args.scramble:
        print(f"  scrambled example: {texts[0][:120]}...")

    print(f"Loading model {args.model_name}")
    model, tokenizer, device = load_model(args.model_name, torch_dtype=torch_dtype)
    print(f"  device={device}, num_hidden_layers="
          f"{model.config.num_hidden_layers}, hidden_size={model.config.hidden_size}")

    print("Encoding...")
    embeddings = encode_all_layers(
        texts, model, tokenizer, device,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(f"  embeddings shape: {embeddings.shape}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(
        output_path,
        embeddings,
        metadata={
            'model_name': args.model_name,
            'pooling': args.pooling,
            'level': args.level,
            'region': args.region,
            'scramble': str(args.scramble),
            'scramble_seed': str(args.scramble_seed) if args.scramble else '',
        },
    )
    print(f"Saved embeddings to {output_path}")


if __name__ == '__main__':
    main()