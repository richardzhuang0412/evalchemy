# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Evalchemy is a unified evaluation toolkit for post-trained language models. It builds on the LM-Eval-Harness and provides a consistent interface for running various benchmarks including instruction-following, reasoning, and code generation tasks. It supports 34+ benchmarks across instruction-following, code generation, and math reasoning domains, with integration for HuggingFace, vLLM, OpenAI, and 100+ API models via Curator.

## Development Setup

### Installation
```bash
conda create --name evalchemy python=3.10
conda activate evalchemy
git clone git@github.com:mlfoundations/evalchemy.git
cd evalchemy
pip install -e .  # Install in editable mode
pip install -e eval/chat_benchmarks/alpaca_eval  # Special dependency
make install  # Runs pip install -e ".[dev]" and pre-commit install
huggingface-cli login  # For gated models/datasets
```

**Notes for HPC systems:**
- May need to modify `pyproject.toml` to use absolute paths for the fschat dependency
- See `eval/distributed/SETUP_*.md` for cluster-specific setup (Capella, Leonardo, TACC, Jureca, etc.)

### Common Development Commands

**Linting and formatting:**
```bash
black --line-length=120 eval/  # Format code
flake8 eval/  # Run linter
```

**Running tests:**
```bash
pytest  # Run all tests in repository
pytest eval/chat_benchmarks/HumanEval/ -v  # Test specific benchmark
pytest eval/chat_benchmarks/MTBench/tests/  # MTBench-specific tests
```

Tests are distributed across benchmark directories (not a centralized `tests/` folder):
- `eval/chat_benchmarks/*/` contain benchmark-specific test files
- MTBench has dedicated `tests/` directory with comprehensive coverage
- Use `--verbose` or `-v` flag for detailed test output

### Running Evaluations

**Basic evaluation:**
```bash
python -m eval.eval \
    --model hf \
    --tasks HumanEval,mmlu \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --batch_size 2 \
    --output_path logs
```

**Using config files** (recommended for standard benchmarks):
```bash
python -m eval.eval \
    --model hf \
    --model_args "pretrained=mistralai/Mistral-7B-Instruct-v0.3" \
    --output_path logs \
    --config configs/light_gpt4omini0718.yaml
```

**With different model backends:**
```bash
# vLLM (high-performance inference)
python -m eval.eval --model vllm --tasks alpaca_eval \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 16 --output_path logs

# OpenAI API
python -m eval.eval --model openai-chat-completions --tasks alpaca_eval \
    --model_args "model=gpt-4o-mini-2024-07-18,num_concurrent=32" \
    --output_path logs

# Curator (100+ API models via LiteLLM)
python -m eval.eval --model curator --tasks AIME24,MATH500 \
    --model_name "gemini/gemini-2.0-flash-thinking-exp-01-21" \
    --apply_chat_template False --output_path logs
```

**Distributed evaluation (HPC):**
```bash
python eval/distributed/launch.py \
    --model_name open-thoughts/OpenThinker-7B \
    --tasks AIME24,AIME25,AMC23,MATH500 \
    --num_shards 8 \
    --watchdog
```

**With database logging:**
```bash
python -m eval.eval \
    --model hf \
    --tasks MTBench,alpaca_eval \
    --model_args 'pretrained=meta-llama/Meta-Llama-3-8B-Instruct' \
    --batch_size 2 \
    --output_path logs \
    --use_database \
    --model_name "My Model Name" \
    --creation_location "Lab Name" \
    --created_by "Researcher Name"
```

**Debug mode** (test on small subset):
```bash
python -m eval.eval \
    --model hf \
    --tasks HumanEval \
    --model_args "pretrained=gpt2" \
    --debug
```

### Viewing and Processing Results

```bash
# View raw results
jq '.results' logs/ModelName/results_timestamp.json

# Extract specific metrics
jq '.results | to_entries[] | {task: .key, metrics: .value}' logs/ModelName/results_timestamp.json

# Convert results to CSV for analysis
python create_csv_helper.py logs/ModelName/results_timestamp.json
```

## Architecture

### Core Components

**1. eval/eval.py** (646 lines) - Main evaluation orchestrator
- Extends LM-Eval-Harness parser with custom arguments: `--database`, `--annotator_model`, `--config`, `--debug`, `--max_tokens`
- Coordinates between pretrain tasks (from lm-eval) and instruction tasks (custom benchmarks)
- Handles result logging to files and optional PostgreSQL database
- Uses `DCEvaluationTracker` for tracking evaluation metadata
- Custom JSON serialization (`handle_non_serializable_extended()`) for large SymPy objects (>15,000 bits)

**2. eval/task.py** (374 lines) - Task management system
- **BaseBenchmark**: Abstract base class for all custom benchmarks with two required methods:
  - `generate_responses(model)`: Creates Instance objects and runs model.generate_until()
  - `evaluate_responses(results)`: Computes metrics from generated outputs
  - `_normalize_model_args(model_args)`: Handles API differences (HF, vLLM, OpenAI, Curator)
  - `_prepare_messages(messages, model)`: Applies chat templates and system instructions
  - `compute(model, instances)`: Batch inference wrapper supporting distributed evaluation
- **TaskManager**: Dynamically loads benchmarks from `eval/chat_benchmarks/`:
  - Scans for directories with `eval_instruct.py`
  - Dynamically imports and instantiates benchmark classes
  - Merges with LM-Eval-Harness pretrain tasks
  - Tracks benchmarks requiring annotator models

**3. eval/chat_benchmarks/** - 34+ custom benchmark implementations
- Each benchmark is a subdirectory with `eval_instruct.py` containing a Benchmark class
- Uses the Instance API to batch generate model outputs efficiently
- Common pattern: load questions → create Instances with prompts → `compute(model, instances)` → evaluate
- Categories:
  - **Instruction Following**: MTBench, WildBench, AlpacaEval, IFEval, RepoBench, MixEval, LiveBench, MBPPPlus
  - **Code Generation**: HumanEval, HumanEvalPlus, MBPP, BigCodeBench, MultiPLE, CRUXEval, LiveCodeBench, CodeForces, CodeElo
  - **Math Reasoning**: AIME24, AIME25, AMC23, MATH500, HMMT, JEEBench, GPQADiamond, Alice in Wonderland
  - **Specialized**: ZeroEval (requires HF approval), SWEbench, MMLUPro, AIW

**4. eval/eval_tracker.py** (419 lines) - Results tracking
- `DCEvaluationTracker`: Persistent result storage and metadata collection
- Logs to JSON files by default
- Optional PostgreSQL integration (enable with `--use_database`)
- Collects: git hash, model config, seeds, tokenizer details, hardware specs, timing

**5. database/** - PostgreSQL schema (SQLAlchemy models)
- `models.py`: Schema definitions (Dataset, Model, EvalResult, EvalSetting)
- `config.py`: Connection configuration (uses env vars: DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_USER)
- `utils.py`: Database utility functions
- Enable with `--use_database` flag and environment variables

**6. eval/distributed/** - HPC cluster evaluation infrastructure
- `launch.py`: Submits SLURM array jobs for data-parallel evaluation
- `process_shard.py`: Worker script processing one data shard per GPU
- Cluster auto-detection via hostname (Capella, Leonardo, TACC, Jureca)
- Workflow: Dataset creation → HF Hub upload → SLURM job submission → Result collection
- Results saved as parquet, uploadable to HuggingFace

### Key Design Patterns

**Instance-based batching:**
Benchmarks create lists of Instance objects which are processed in batches by the model:
```python
from lm_eval.api.instance import Instance

instances = [
    Instance(
        "generate_until",
        example,
        (templated_prompt, {"max_new_tokens": 1024, "do_sample": False}),
        idx
    )
]
outputs = self.compute(model, instances)
```

**Chat template handling:**
The `_prepare_messages()` method in BaseBenchmark handles system instructions and applies chat templates:
```python
messages = [{"role": "user", "content": prompt}]
templated = self._prepare_messages(messages, model)
```

**Model arguments normalization:**
`_normalize_model_args()` in BaseBenchmark handles differences between HuggingFace, vLLM, and OpenAI APIs:
- **HuggingFace**: Uses `max_new_tokens`, no seed support
- **vLLM**: Uses `max_gen_toks`, supports seed
- **OpenAI**: Uses `max_tokens`, caps gpt-4o at 16384 tokens
- **Curator**: Wrapper for 100+ models via LiteLLM

**Dynamic benchmark discovery:**
TaskManager uses runtime introspection to load benchmarks with no central registry:
- Easy to add new benchmarks (create directory + `eval_instruct.py`)
- Benchmarks are self-contained with data

### Database Integration

Optional PostgreSQL integration for leaderboard and experiment tracking:
- Enable with `--use_database` flag when running evaluations
- Configure via environment variables: `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`
- Schema tables: `models`, `evalresults`, `evalsettings`, `datasets`
- Results linked to model UUIDs for tracking experiments over time
- Use `--model_id` or `--model_name` to update existing entries
- See `database/README.md` for schema details

### Distributed Evaluation

For large-scale evaluations across HPC clusters:
- **Location**: `eval/distributed/`
- **Main scripts**: `launch.py` (submit SLURM jobs), `process_shard.py` (worker script)
- **Workflow**: Create dataset → Upload to HF Hub → Submit array job → Collect results → Compute metrics
- **Cluster auto-detection**: Reads hostname to detect cluster (Capella, Leonardo, TACC, Jureca)
- **Output**: Results saved as parquet files, uploadable to HuggingFace
- **Setup guides**: `eval/distributed/SETUP_*.md` for each supported cluster

### Config Files

YAML configurations in `configs/` directory enable standardized evaluation runs:
- **light_gpt4omini0718.yaml**: Lightweight benchmarks with GPT-4o Mini judge
- **full_gpt4omini0718.yaml**: Full benchmark suite
- **reasoning.yaml / reasoning_lite.yaml**: Math and reasoning tasks
- **single_task/**: Individual task configurations
- Config format:
  ```yaml
  annotator_model: gpt-4o-mini-2024-07-18
  tasks:
    - task_name: alpaca_eval
      batch_size: 32
    - task_name: WildBench
      batch_size: 8
  max_tokens: 1024
  ```
- Configs override command-line `--batch_size`, `--tasks`, `--annotator_model`, `--max_tokens`

## Adding New Benchmarks

**Standard benchmark structure:**
1. Create directory in `eval/chat_benchmarks/YourBenchmark/`
2. Create `eval_instruct.py` with a class inheriting from `BaseBenchmark`:
   ```python
   from eval.task import BaseBenchmark
   from lm_eval.api.instance import Instance
   from typing import Dict, Any

   class YourBenchmarkBenchmark(BaseBenchmark):
       def generate_responses(self, model) -> Dict[str, Any]:
           # Load dataset and create instances
           instances = [
               Instance("generate_until", example, (prompt, stop_seq), idx)
               for idx, example in enumerate(dataset)
           ]
           # Run batch inference
           results = self.compute(model, instances)
           return {"outputs": results, "references": references}

       def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
           # Compute metrics from results
           outputs = results["outputs"]
           references = results["references"]
           # Calculate accuracy, F1, etc.
           return {"accuracy": score}
   ```
3. Add benchmark class name to `__init__.py` if needed (TaskManager auto-discovers via `eval_instruct.py`)
4. Add tested metrics to `reproduced_benchmarks.md`
5. Add benchmark entry to `README.md`

**Key patterns for benchmark implementation:**
- Use `_prepare_messages()` for chat template handling
- Use `_normalize_model_args()` for API-agnostic model arguments
- Use `self.compute(model, instances)` for batch inference (supports distributed evaluation)
- Store raw outputs and references in results dict for downstream analysis
- Return flat dict of metric names to float values

## Important Notes

### Environment & Authentication
- Set `OPENAI_API_KEY` for LLM judge models (e.g., gpt-4o-mini)
- Run `huggingface-cli login` for accessing gated models/datasets
- May need `ANTHROPIC_API_KEY` for Claude models, `MISTRAL_API_KEY` for Mistral, etc.

### Special Dependencies & Requirements
- **BigCodeBench**: Requires Docker for safe code execution
- **ZeroEval**: Requires HuggingFace dataset access approval
- **MTBench**: Installed as git subtree from `eval/chat_benchmarks/MTBench`
- **alpaca_eval**: Installed as git subtree, requires separate pip install

### Metadata & Results
- All evaluation results include extensive metadata: model config, seeds, git hash, hardware specs (GPU, CUDA), timing
- Custom JSON serialization handles large SymPy objects (>15,000 bits) via `handle_non_serializable_extended()`
- Results organized as `logs/ModelName/results_timestamp.json`

### Model Compatibility
- **Supported model backends**: HuggingFace (hf), vLLM, OpenAI, Curator (LiteLLM wrapper for 100+ models)
- **Chat template support**: Essential for instruction-following benchmarks; auto-applied via `_prepare_messages()`
- **Seed handling**: Different APIs handle seeds differently; `_normalize_model_args()` abstracts this
