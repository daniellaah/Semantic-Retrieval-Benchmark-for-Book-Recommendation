#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

EVAL_INPUT="${EVAL_INPUT:-data/processed/eval_u6_i5_q5.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/eval/last}"
TOPK="${TOPK:-10,50,100}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
SEED="${SEED:-42}"
MAX_QUERY="${MAX_QUERY:-0}"

if [[ ! -f "$EVAL_INPUT" ]]; then
  echo "Missing eval input: $EVAL_INPUT" >&2
  exit 1
fi

if [[ ! -d "outputs/embeddings" ]]; then
  echo "Missing embeddings directory: outputs/embeddings" >&2
  exit 1
fi

BATCH_TS="$(date +"%Y%m%d%H%M%S")"

mkdir -p "$OUTPUT_ROOT"

find "outputs/embeddings" -mindepth 2 -maxdepth 2 -type d | sort | while read -r embedding_dir; do
  if [[ ! -f "$embedding_dir/item_ids.jsonl" ]]; then
    echo "Skipping $embedding_dir: missing item_ids.jsonl"
    continue
  fi

  model_dir="$(basename "$(dirname "$embedding_dir")")"
  embedding_run_id="$(basename "$embedding_dir")"
  eval_run_id="${BATCH_TS}_${model_dir}_${embedding_run_id}"

  echo "============================================================"
  echo "embedding_dir=$embedding_dir"
  echo "eval_run_id=$eval_run_id"

  KMP_DUPLICATE_LIB_OK=TRUE UV_CACHE_DIR=.uv-cache uv run python src/eval/run_eval.py \
    --eval-input "$EVAL_INPUT" \
    --embedding-dir "$embedding_dir" \
    --embedding-dim all \
    --output-root "$OUTPUT_ROOT" \
    --eval-run-id "$eval_run_id" \
    --query-pooling last \
    --topk "$TOPK" \
    --index-type "$INDEX_TYPE" \
    --seed "$SEED" \
    --max-query "$MAX_QUERY"
done
