"""
TwoNN intrinsic dimensionality estimation (Facco et al., 2017).
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


def estimate_id_twonn(X):
    """MLE of intrinsic dim from the ratio of 1st/2nd nearest-neighbor distances."""
    dists = squareform(pdist(X, metric='euclidean'))

    # exclude self-distances
    np.fill_diagonal(dists, np.inf)
    sorted_dists = np.sort(dists, axis=1)
    r1 = sorted_dists[:, 0]
    r2 = sorted_dists[:, 1]

    # drop duplicates where r1 == 0
    valid = r1 > 0
    mu = r2[valid] / r1[valid]

    if len(mu) == 0:
        return 0.0

    # pareto MLE: d = n / sum(log(mu))
    return len(mu) / np.sum(np.log(mu))


def compute_id_per_layer(layer_embeddings):
    """I_d at each layer. Input shape (n_layers, n_samples, hidden_dim)."""
    n_layers = layer_embeddings.shape[0]
    id_values = np.empty(n_layers)
    for i in range(n_layers):
        id_values[i] = estimate_id_twonn(layer_embeddings[i])
    return id_values


def save_id_results(path, id_values, metadata=None):
    to_save = {'id_values': id_values}
    if metadata:
        for k, v in metadata.items():
            to_save[k] = np.asarray(v)
    np.savez_compressed(path, **to_save)


def load_id_results(path):
    data = np.load(path, allow_pickle=True)
    id_values = data['id_values']
    metadata = {k: data[k] for k in data.files if k != 'id_values'}
    return id_values, metadata
