#!/bin/bash

# python -m driver.rag \
#   --prefixes ./data/prefixes.jsonl \
#   --output_dir ./data/

  python -m retriever.driver.rag \
  --prefixes ./data/prefixes.jsonl \
  --output_dir ./data/