Code for project "Evaluating the Brain Predictivity of Sentence Embeddings: A Comparison Between Large Language Models and Text Embedders". Main code is within the scripts folder. extract_embeddings.py takes a passage, extracts representational embeddings and saves it. Then the next script, run_encoding.py fits an encoding model to these representations. These scripts are called using the files in the jobs folder, and visualized using 'visualization/visualization.ipynb'.

### Data
```bash
bash scripts/download_data.sh 
```
This downloads the Pereira et al (2018) fMRI dataset into 'MRI_data/<participant>'.


### Environment
The job scripts activate a venv at `$HOME/venvs/nlp2brain`. To create it locally:
```bash
python -m venv ~/venvs/nlp2brain
source ~/venvs/nlp2brain/bin/activate
pip install numpy scipy scikit-learn himalaya torch transformers
```

## Reproducing Results (Snellius, SLURM)
We made use of the job scripts in the jobs folder. After downloading the data. Run:
```bash
sbatch Participant_normal_results_experiment2.job   # Encoding — 384-sentence dataset
sbatch Participant_normal_results_experiment3.job   # Encoding — 243-sentence dataset
sbatch Participant_scramble_exp2.job               # Bag-of-words ablation, exp 2
sbatch Participant_scramble_exp3.job               # Bag-of-words ablation, exp 3
sbatch Participant_all_regions.job                 # Encoding across all brain regions
sbatch intrinsic_dim.job                           # Intrinsic dimensionality of embeddings
```

## Single participant
To run obtain results for one participant, chain the following two scripts.
scripts/extract_embeddings.py which extract passages, after which representations of all layers of an language model are extracted and saved to a data folder so that it only needs to be ran once and easily read from afterwards.
Followed by scripts/run_encoding.py which fits an encoding model to these representations and saves the results voxel-wise.
Example usage:

```bash
# 1. Extract layer-wise embeddings
python scripts/extract_embeddings.py \
    --mat-path MRI_data/P01/data_384sentences.mat \
    --model-name Qwen/Qwen3-4B \
    --pooling last \
    --output data/experiment2/P01_qwen3-4b-llm_passages.npz \
    --level passage --dtype float16

# 2. Fit encoding models and save voxel-wise results
python scripts/run_encoding.py \
    --mat-path MRI_data/P01/data_384sentences.mat \
    --embeddings data/experiment2/P01_qwen3-4b-llm_passages.npz \
    --output-dir results/experiment2/P01_qwen3-4b-llm_passages \
    --level passage --save-voxelwise
```
Swap `--model-name Qwen/Qwen3-Embedding-4B` for the embedding model. Participants by experiment:
- **Experiment 2** (`data_384sentences.mat`): P01 M02 M04 M07 M08 M09 M14 M15  
- **Experiment 3** (`data_243sentences.mat`): P01 M02 M03 M04 M07 M15

## Visualization
Once all results are generated, they can be visualized using 'visualization/visualization.ipynb'.
