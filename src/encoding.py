"""
Voxel-wise encoding models.

Fits kernel ridge regression from model embeddings to brain responses,
with per-voxel alpha selection (via himalaya's KernelRidgeCV).

Logic is wrapped in reusable functions that can be called to fit a layer.
"""

import numpy as np
from himalaya.kernel_ridge import KernelRidgeCV
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Default alphas for the CV search - matches Project C.
DEFAULT_ALPHAS = np.logspace(1, 20, 20)


def fit_encoding_model(model_embeddings, brain_responses,
                       n_splits=5, alphas=None, inner_cv_splits=5,
                       verbose=False):
    """
    Fit a voxel-wise encoding model and return per-fold accuracies.

    This is the exact pipeline from Project C: K-fold CV on the outer loop,
    StandardScaler (centering only) + KernelRidgeCV inside, with per-voxel
    alpha selection via an inner 5-fold CV. Accuracy is per-voxel Pearson
    correlation between predictions and true responses.

    Args:
        model_embeddings: (n_stimuli, n_features) array.
        brain_responses:  (n_stimuli, n_voxels) array.
        n_splits: outer KFold splits.
        alphas: array of alpha values for the inner CV search.
        inner_cv_splits: number of inner CV folds for alpha selection.
        verbose: if True, print per-fold means.

    Returns:
        dict with 'accs_train' and 'accs_test', each shape (n_splits, n_voxels).
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    kf = KFold(n_splits=n_splits)
    _, n_voxels = brain_responses.shape

    accs_train = np.empty((n_splits, n_voxels))
    accs_test = np.empty((n_splits, n_voxels))

    for i, (train_index, test_index) in enumerate(kf.split(model_embeddings)):

        X_train = model_embeddings[train_index, :]
        X_test = model_embeddings[test_index, :]
        Y_train = brain_responses[train_index, :]
        Y_test = brain_responses[test_index, :]

        pipeline = make_pipeline(
            StandardScaler(with_mean=True, with_std=False),
            KernelRidgeCV(alphas=alphas, cv=KFold(n_splits=inner_cv_splits))
        )
        pipeline.fit(X_train, Y_train)

        preds_train = pipeline.predict(X_train)
        corrs_train, _ = pearsonr(preds_train, Y_train, axis=0)
        accs_train[i] = corrs_train

        preds_test = pipeline.predict(X_test)
        corrs_test, _ = pearsonr(preds_test, Y_test, axis=0)
        accs_test[i] = corrs_test

        if verbose:
            print(f"  fold {i}: train {corrs_train.mean():.3f}  "
                  f"test {corrs_test.mean():.3f}")

    return {'accs_train': accs_train, 'accs_test': accs_test}


def fit_encoding_per_layer(layer_embeddings, brain_responses,
                           n_splits=5, alphas=None, inner_cv_splits=5,
                           verbose=True):
    """
    Run fit_encoding_model once per layer.

    Args:
        layer_embeddings: (n_layers, n_stimuli, hidden_dim) array.
        brain_responses:  (n_stimuli, n_voxels) array.

    Returns:
        dict with 'accs_train' and 'accs_test', each shape
        (n_layers, n_splits, n_voxels).
    """
    n_layers = layer_embeddings.shape[0]
    _, n_voxels = brain_responses.shape

    all_accs_train = np.empty((n_layers, n_splits, n_voxels))
    all_accs_test = np.empty((n_layers, n_splits, n_voxels))

    for layer_idx in range(n_layers):
        if verbose:
            print(f"Fitting layer {layer_idx}/{n_layers - 1}")

        result = fit_encoding_model(
            layer_embeddings[layer_idx],
            brain_responses,
            n_splits=n_splits,
            alphas=alphas,
            inner_cv_splits=inner_cv_splits,
            verbose=verbose,
        )
        all_accs_train[layer_idx] = result['accs_train']
        all_accs_test[layer_idx] = result['accs_test']

        if verbose:
            print(f"  layer {layer_idx} mean test acc: "
                  f"{all_accs_test[layer_idx].mean():.3f}")

    return {'accs_train': all_accs_train, 'accs_test': all_accs_test}


def summarize_layer_accuracies(accs):
    """
    Collapse per-voxel, per-fold accuracies down to one number per layer.

    Args:
        accs: (n_layers, n_splits, n_voxels) array.

    Returns:
        (n_layers,) array of mean accuracy per layer.
    """
    return accs.mean(axis=(1, 2))