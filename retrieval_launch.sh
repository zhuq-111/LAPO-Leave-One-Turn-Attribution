#!/usr/bin/env bash
# Copyright 2026 LOTAPO Authors
# Licensed under the Apache License, Version 2.0. See LICENSE in the project root.

set -euo pipefail

: "${INDEX_PATH:?Set INDEX_PATH to the E5 FAISS index}"
: "${CORPUS_PATH:?Set CORPUS_PATH to the Wikipedia JSONL corpus}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m lotapo.search.retrieval_server \
  --index_path "$INDEX_PATH" \
  --corpus_path "$CORPUS_PATH" \
  --topk 3 \
  --retriever_name "e5" \
  --retriever_model "${RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
  --faiss_gpu \
  --retrieval_batch_size "${RETRIEVAL_BATCH_SIZE:-16}"
