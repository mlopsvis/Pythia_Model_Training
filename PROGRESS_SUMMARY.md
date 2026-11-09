# HW8 Progress Summary — Train a Retriever

## What was accomplished

### 1. Implemented the retriever encoder (`src/retriever/modeling/encoder.py`)
Completed all 4 previously-stubbed methods:
- `pooling` — last-token pooling + L2 normalize
- `compute_similarity` — dot product scaled by temperature
- `compute_labels` — positive-passage index targets (`arange * group_size`)
- `compute_loss` — cross-entropy (InfoNCE with in-batch negatives)

### 2. Architecture clarified (two separate models)
- **Pythia-160M** — the retriever you *train* (used by encode → search → RAG retrieval).
- **Pythia-6.9B** (`allenai/open-instruct-pythia-6.9b-tulu`) — the generator in `rag.py`.
- `generate_without_rag.sh` — baseline (6.9B only, 160M NOT used).
- `generate_with_rag.sh` — uses your 160M retriever's docs (needs `get_rag`, still unimplemented).

### 3. Training made Spot-instance resilient (`src/retriever/scripts/train_msmarco.sh`)
- Added `--save_strategy steps --save_steps 0.2 --save_total_limit 5` (checkpoints every 20%).
- Added bash auto-resume: detects existing `checkpoint-*` → passes `--resume_from_checkpoint true`.
- No dataset changes needed (auto-downloads `Tevatron/msmarco-passage-aug`, slices 100k).
- A100 80GB ready: flash-attn2 ✅, bf16 ✅.

### 4. GCP VM / disk guidance
- `nvme0n1` (375G) is an **ephemeral Local SSD** — must NOT store checkpoints.
- Boot disk `/` (sda1) is persistent, survives Spot *stop* — default `./data/model` path is fine.
- Key check: Spot termination action must be `STOP`, not `DELETE`.

### 5. Git + GitHub
- Leaked PAT in chat was **revoked**.
- Repo initialized; `.gitignore` extended (excludes weights / embeddings / wandb / submission.zip).
- Committed 35 files (code only, no secrets/weights) and pushed to
  `https://github.com/mlopsvis/Pythia_Model_Training.git` — verified on remote (`main` @ `c5542b3`).

## Current state
- ✅ Encoder implemented, code on GitHub, training script Spot-ready.
- ⏳ Next: clone on VM, finish README env setup (only `wandb` done so far), run training in `tmux`.

## Still pending (later homework steps)
- `get_rag` + corpus parsing in `src/retriever/driver/rag.py` (for the with-RAG path).
- Run encode → search → generate-with-rag pipeline using the trained 160M model.
- Keep the trained 160M weights for the next homework.

## Immediate next commands (on the VM)
```bash
git clone https://github.com/mlopsvis/Pythia_Model_Training.git hw8 && cd hw8
# conda env + pip install -r + faiss-gpu + flash-attn + pip install -e .
cd src/retriever && tmux new -s train
bash scripts/train_msmarco.sh 2>&1 | tee train.log
```
