#!/bin/bash
#SBATCH --job-name=depthbench-tokenize
#SBATCH --account=davisjam
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tokenize-%j.out
#SBATCH --error=logs/tokenize-%j.err

# Pre-tokenize FineWeb-Edu 100BT into data/fineweb-edu/pre-tokenize/{train,eval}/*.npy.
# CPU-only work; the GPU is requested only because every Gilbreth partition requires one.
# One process per CPU, each handling every NW-th parquet file; worker 0 also does the eval shard.
# Shards are written atomically, so resubmitting resumes via --skip-existing.
# Submit from the repository root: sbatch data_utils/tokenize_fineweb.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$PWD/pretrain/OLMo-core/src:$PWD"
PY="$PWD/.venv-cu128/bin/python"
NW=${SLURM_CPUS_PER_TASK:-32}

common_args=(
    --train-parquet-glob "data/fineweb-edu/100BT/*.parquet"
    --eval-parquet-path "data/fineweb-edu/eval/eval_013_00008.parquet"
    --output-dir "data/fineweb-edu/pre-tokenize"
    --text-field "text"
    --tokenizer-name-or-path "pretrain/OLMo-core/src/olmo_core/data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"
    --vocab-size 50280 --eos-token-id 50279 --pad-token-id 1
    --batch-size 4096
    --write-doc-indices --skip-existing
)

pids=()
for ((w = 0; w < NW; w++)); do
    extra=(--skip-summary)
    ((w > 0)) && extra+=(--skip-eval)
    "$PY" data_utils/tokenize_from_pretrain_datasets.py "${common_args[@]}" "${extra[@]}" \
        --train-worker-id "$w" --train-num-workers "$NW" \
        > "logs/tokenize-${SLURM_JOB_ID}-w${w}.log" 2>&1 &
    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
if ((failed)); then
    echo "At least one worker failed; see logs/tokenize-${SLURM_JOB_ID}-w*.log" >&2
    exit 1
fi

# Collect every shard's metadata into manifest.jsonl / summary.json.
"$PY" data_utils/tokenize_from_pretrain_datasets.py "${common_args[@]}" --skip-train --skip-eval
cat data/fineweb-edu/pre-tokenize/summary.json
