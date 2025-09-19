# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

export WANDB_PROJECT="litgpt-test"
export WANDB_RUN_NAME="lora-qwen3-8b-alpaca"
export WANDB_MODE="offline"
# export CUDA_VISIBLE_DEVICES=0

litgpt finetune_lora Qwen/Qwen3-8B \
    --data TinyStories \
    --devices 1 \
    --num_nodes 1 \
    --train.global_batch_size 16 \
    --train.micro_batch_size 1 \
    --eval.interval 10000 \
    --seed 42 \
    --logger_name wandb \
    --lora_r 32 \
    --lora_query True \
    --lora_key True \
    --lora_value True \
    --lora_head True \
    --lora_projection True \
    --lora_mlp True \
    --train.max_seq_length 2048 \
    --precision bf16-true \