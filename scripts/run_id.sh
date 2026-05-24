#!/bin/bash
set -e

echo "================================"
echo "  Intrinsic Dimensionality Pipeline"
echo "================================"
echo ""

export HF_HOME=/scratch-shared/$USER/hf_dir
export HF_HUB_CACHE=/scratch-shared/$USER/hf_dir
mkdir -p data/experiment2 data/experiment3 results/intrinsic_dim

TOTAL=4
CURRENT=0

for EXP in experiment2 experiment3; do
    if [ "$EXP" = "experiment2" ]; then
        MAT=MRI_data/M02/data_384sentences.mat
    else
        MAT=MRI_data/M02/data_243sentences.mat
    fi

    for MODEL_KEY in llm embedding; do
        CURRENT=$((CURRENT + 1))

        if [ "$MODEL_KEY" = "llm" ]; then
            MODEL_NAME=Qwen/Qwen3-4B
        else
            MODEL_NAME=Qwen/Qwen3-Embedding-4B
        fi

        TAG=qwen3-4b-${MODEL_KEY}_passages
        EMB=data/${EXP}/${TAG}.npz

        echo ""
        echo "[$CURRENT/$TOTAL] $MODEL_KEY — $EXP"
        echo "────────────────────────────────"

        if [ ! -f "$EMB" ]; then
            echo "  → extracting embeddings ($MODEL_NAME)..."
            python scripts/extract_embeddings.py \
                --mat-path $MAT \
                --model-name $MODEL_NAME \
                --pooling last \
                --output $EMB \
                --level passage --dtype float16
            echo "  ✓ embeddings saved to $EMB"
        else
            echo "  ✓ embeddings already exist, skipping"
        fi

        echo "  → computing I_d..."
        python scripts/compute_intrinsic_dim.py \
            --embeddings $EMB \
            --output-dir results/intrinsic_dim/${EXP}/${TAG}
        echo "  ✓ done"
    done
done

echo ""
echo "================================"
echo "  All done!"
echo "  Results: results/intrinsic_dim/"
echo "================================"
echo ""
echo "Summary:"
for f in results/intrinsic_dim/*/qwen3-4b-*/intrinsic_dim_summary.txt; do
    echo "  $f"
    head -1 "$f"
    awk 'NR>1 {if($2+0 > max) {max=$2; layer=$1}} END {printf "    peak I_d: layer %s (%.2f)\n", layer, max}' "$f"
done
