source .venv/bin/activate

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

export WANDB_PROJECT="litgpt-test"
export WANDB_RUN_NAME="ft-qwen3-8b-alpaca"
export WANDB_MODE="offline"

litgpt finetune_full Qwen/Qwen3-0.6B \
    --data Alpaca \
    --logger_name wandb \
    --devices 1 \
    --num_nodes 1 \
    --seed 42 \
    --train.global_batch_size 16 \
    --train.micro_batch_size 1 \
    --train.max_seq_length 256 \
    --precision bf16-true \

exit 0