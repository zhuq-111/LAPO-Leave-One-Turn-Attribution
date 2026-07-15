# LAPO

This repository implements **LAPO: Leave-One-Turn Attribution for Self-Generated
Process Rewards in Multi-Turn Search Reasoning** on top of VERL/Search-R1.

## Algorithm-to-code map

```text
LAPO/
├── lapo/
│   ├── llm_agent/
│   │   └── generation.py
│   │       └── LLMGenerationManager
│   │           ├── Multi-turn search rollout
│   │           ├── _score_lapo_rows
│   │           │   └── Gold-answer mean log-likelihood
│   │           └── _compute_counterfactual_ig_rewards
│   │               └── Backward [DELETE] counterfactual
│   ├── algorithm.py
│   │   ├── F1 terminal reward and GRPO outcome advantage
│   │   └── Robust scaling, tanh, group normalization, and sign gate
│   └── prompts.py
│       └── Paper prompt templates
└── verl/
    └── trainer/
        ├── main_ppo.py
        │   ├── F1 terminal reward and GRPO outcome advantage
        │   └── Token-level outcome + process advantage
        └── ppo/
            └── core_algos.py
                └── Clipped GRPO optimization
```

The numerical LAPO equations are isolated in `lapo/algorithm.py`; they do not import
Ray, Torch, or trainer code and can be unit-tested independently.

## Training environment

The reference training environment was captured on Linux x86_64 with Python 3.9.25,
PyTorch 2.4.0 (CUDA 12.1), vLLM 0.6.3, and NVIDIA A100 GPUs. Create it with:

```bash
conda env create -f environment.yml
conda activate lapo

# Install PyTorch first so flash-attn can compile against the active Torch/CUDA ABI.
python -m pip install torch==2.4.0 torchvision==0.19.0
python -m pip install flash-attn==2.8.2 --no-build-isolation
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` lists pinned direct dependencies for the training environment.
The source server used a prebuilt `flash-attn` wheel for Python 3.9, Torch 2.4, and CUDA 12;
using a matching wheel is preferable when one is available.

The retrieval server is a separate environment: the captured training environment
does not contain FAISS. Its dependencies must be installed independently before
running `retrieval_launch.sh`.

## Data

The checked-in `data/data_4full` files contain the paper-format prompts. To rebuild the
four-dataset training split and seven-dataset evaluation split:

```bash
bash scripts/nq_hotpotqa/data_process.sh
```

The training data sources are NQ, TriviaQA, HotpotQA, and 2WikiMultiHopQA. Each source
contributes 20,000 training examples in the paper; apply that sampling policy when
preparing a fresh release dataset.


## Training

The main script exposes paths through environment variables and otherwise uses the paper
configuration: Qwen2.5-3B-Instruct, five trajectories per question, three searches and sign-consistency gating.

```bash
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
DATA_DIR=data/data_4full \
bash train_LAPO.sh
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover group-level robust scaling, eligible-turn masking, sign-consistency
gating, all-zero attribution, and retention of the terminal outcome advantage.

## Before release

Add a pinned environment or lock file before publishing.

## License

Copyright 2026 LAPO Authors.

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
for the license terms and [NOTICE](NOTICE) for attribution of upstream software.
