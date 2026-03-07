#!/usr/bin/env bash

UV_CACHE_DIR=.uv-cache uv run python src/embedding/generate_item_embeddings.py \
  --experiment-config configs/tac/{config} \
  --items-input data/processed/items_subset_eval_u6_i5_q5.jsonl \
  --device mps \
  --batch-size 64

UV_CACHE_DIR=.uv-cache uv run python src/eval/run_eval.py \
  --eval-input data/processed/eval_u6_i5_q5.jsonl \
  --embedding-dir outputs/embeddings/{embeddings} \
  --embedding-dim all \
  --output-root outputs/eval/last \
  --topk 10,50,100 \
  --index-type hnsw

XDG_CACHE_HOME=.cache \
MPLCONFIGDIR=.cache/matplotlib \
KMP_DUPLICATE_LIB_OK=TRUE \
UV_CACHE_DIR=.uv-cache \
uv run python src/eval/plot_eval_results.py \
    --input outputs/eval/last \
    --output-dir outputs/eval/last/plots \
