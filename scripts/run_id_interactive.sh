#!/bin/bash
# run this in an interactive session on snellius
# usage: bash scripts/run_id_interactive.sh

set -e

cd /home/scur0418/NLP2BrainEmbed

module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source $HOME/venvs/nlp2brain/bin/activate

export HF_HOME=/scratch-shared/$USER/hf_dir
export HF_HUB_CACHE=/scratch-shared/$USER/hf_dir

mkdir -p MRI_data data/experiment2 data/experiment3 results/intrinsic_dim

# download M02 if needed
if [ ! -d MRI_data/M02 ]; then
    echo "downloading M02 data..."
    wget -q -O MRI_data/M02.tar "https://www.dropbox.com/s/n5yfb2cupd9zmwk/M02.tar?dl=1"
    tar xf MRI_data/M02.tar -C MRI_data/
    rm MRI_data/M02.tar
    echo "done"
fi

for EXP in experiment2 experiment3; do
    if [ "$EXP" = "experiment2" ]; then
        MAT=MRI_data/M02/data_384sentences.mat
    else
        MAT=MRI_data/M02/data_243sentences.mat
    fi

    for MODEL_KEY in llm embedding; do
        if [ "$MODEL_KEY" = "llm" ]; then
            MODEL_NAME=Qwen/Qwen3-4B
        else
            MODEL_NAME=Qwen/Qwen3-Embedding-4B
        fi

        TAG=qwen3-4b-${MODEL_KEY}_passages
        EMB=data/${EXP}/${TAG}.npz

        if [ ! -f "$EMB" ]; then
            echo "=== extracting ${TAG} (${EXP}) ==="
            python scripts/extract_embeddings.py \
                --mat-path $MAT \
                --model-name $MODEL_NAME \
                --pooling last \
                --output $EMB \
                --level passage --dtype float16
        else
            echo "=== ${TAG} (${EXP}) already exists, skipping ==="
        fi

        echo "=== computing I_d for ${TAG} (${EXP}) ==="
        python scripts/compute_intrinsic_dim.py \
            --embeddings $EMB \
            --output-dir results/intrinsic_dim/${EXP}/${TAG}
    done
done

echo ""
echo "all done! results in results/intrinsic_dim/"
