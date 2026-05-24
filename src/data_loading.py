"""
Data loading for the Pereira fMRI dataset.

Does the following:
- Loading the .mat files for a participant (These are the sentences)
- Isolating a chosen atlas/ROI's voxels (e.g. languageLH. The brain region we're looking at.)
- Building passage-level text by joining their constituent sentences
- Optional bag-of-words ablation (scramble word order within each stimulus)
"""

import random

import numpy as np
from scipy.io import loadmat


# Regions available in the Pereira dataset's `meta.atlases`.
AVAILABLE_REGIONS = [
    'aal_1ofN',
    'languageParcelsConservative_aal',
    'languageParcels_aal',
    'semantic_aal',
    'multipleDemand_aal',
    'MD',
    'DMN',
    'languageLH',
    'languageRH',
    'visual_body',
    'visual_face',
    'visual_object',
    'visual_scene',
    'visual',
    'gordon',
]


def load_participant_data(mat_path):
    """
    Load a participant's .mat file (e.g., M02/data_243sentences.mat).
    """
    data = loadmat(mat_path, simplify_cells=True)
    return data


def get_region_column_indexes(data, region='languageLH'):
    """
    Return the voxel/column indexes corresponding to a given atlas region.
    """
    atlases = data['meta']['atlases']
    matches = np.where(atlases == region)[0]
    if matches.size == 0:
        available = list(atlases)
        raise ValueError(
            f"Region {region!r} not found in data['meta']['atlases']. "
            f"Available regions: {available}"
        )
    target_index = matches.item()
    column_indexes = np.concatenate(
        [arr - 1 for arr in data['meta']['roiColumns'][target_index]],
        axis=0
    )
    return column_indexes


def get_passage_brain_responses(data, column_indexes):
    return data['examples_passages'][:, column_indexes]


def get_sentence_brain_responses(data, column_indexes):
    return data['examples_passagesentences'][:, column_indexes]


def _scramble(text, rng):
    """
    Shuffle whitespace-separated tokens.
    """
    tokens = text.split()
    rng.shuffle(tokens)
    return ' '.join(tokens)


def get_passages(data, scramble=False, scramble_seed=42):
    """
    Build passage-level texts by joining their constituent sentences.
    """
    n_passages = int(data['labelsPassageForEachSentence'].max())
    passages = [
        ' '.join(data['keySentences'][data['labelsPassageForEachSentence'] == p])
        for p in range(1, n_passages + 1)
    ]
    if scramble:
        passages = [
            _scramble(p, random.Random(scramble_seed + i))
            for i, p in enumerate(passages)
        ]
    return passages


def get_sentences(data, scramble=False, scramble_seed=42):
    """
    Return the list of sentence-level stimuli.
    """
    sentences = list(data['keySentences'])
    if scramble:
        sentences = [
            _scramble(s, random.Random(scramble_seed + i))
            for i, s in enumerate(sentences)
        ]
    return sentences


def load_stimuli_and_responses(
    mat_path,
    level='passage',
    region='languageLH',
    scramble=False,
    scramble_seed=42,
):
    data = load_participant_data(mat_path)
    column_indexes = get_region_column_indexes(data, region=region)

    if level == 'passage':
        texts = get_passages(data, scramble=scramble, scramble_seed=scramble_seed)
        brain_responses = get_passage_brain_responses(data, column_indexes)
    elif level == 'sentence':
        texts = get_sentences(data, scramble=scramble, scramble_seed=scramble_seed)
        brain_responses = get_sentence_brain_responses(data, column_indexes)
    else:
        raise ValueError(f"level must be 'passage' or 'sentence', got {level!r}")

    return texts, brain_responses