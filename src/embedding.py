"""
Model embedding extraction.

Uses the raw HuggingFace transformers API (rather than SentenceTransformer. Which is easier but less flexibile as we're interested in all layers)
so that we can request output_hidden_states=True and get one embedding per
layer per stimulus. 

For a model with N transformer layers, outputs['hidden_states'] is a tuple
of length N+1 (the +1 is the embedding layer before any transformer block).
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_model(model_name, device=None, torch_dtype=None):
    """
    Load a HuggingFace model and tokenizer.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {}
    if torch_dtype is not None:
        kwargs['torch_dtype'] = torch_dtype

    model = AutoModel.from_pretrained(model_name, **kwargs).to(device)
    model.eval()

    return model, tokenizer, device


def _pool_hidden_state(hidden_state, attention_mask, pooling):
    """
    Collapse the sequence dimension of a single layer's hidden state into
    a single vector per item in the batch.
    """
    if pooling == 'cls':
        return hidden_state[:, 0, :]

    if pooling == 'last':
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_state.size(0), device=hidden_state.device)
        return hidden_state[batch_idx, last_idx, :]

    if pooling == 'mean':
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        summed = (hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    raise ValueError(f"Unknown pooling {pooling!r}. Use 'last', 'cls', or 'mean'.")


def encode_all_layers(texts, model, tokenizer, device,
                      pooling='last', batch_size=1, max_length=None):
    """
    Encode a list of texts with a transformer model, returning one embedding
    per layer per text.
    """
    num_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size
    num_texts = len(texts)

    all_embeddings = np.zeros((num_layers, num_texts, hidden_dim), dtype=np.float32)

    tokenize_kwargs = dict(padding=True, truncation=True, return_tensors='pt')
    if max_length is not None:
        tokenize_kwargs['max_length'] = max_length

    for start in range(0, num_texts, batch_size):
        end = min(start + batch_size, num_texts)
        batch_texts = texts[start:end]

        inputs = tokenizer(batch_texts, **tokenize_kwargs).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        attention_mask = inputs['attention_mask']

        for layer_idx, hs in enumerate(outputs.hidden_states):
            pooled = _pool_hidden_state(hs, attention_mask, pooling)
            all_embeddings[layer_idx, start:end] = pooled.float().cpu().numpy()

        del outputs, inputs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return all_embeddings


def save_embeddings(path, embeddings, metadata=None):
    """
    Save layer-wise embeddings to disk as .npz.
    """
    to_save = {'embeddings': embeddings}
    if metadata:
        for k, v in metadata.items():
            to_save[k] = np.asarray(v)
    np.savez_compressed(path, **to_save)


def load_embeddings(path):
    """
    Load embeddings saved by save_embeddings.
    """
    data = np.load(path, allow_pickle=True)
    embeddings = data['embeddings']
    metadata = {k: data[k] for k in data.files if k != 'embeddings'}
    return embeddings, metadata