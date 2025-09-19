# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

export WANDB_PROJECT="litgpt-test"
# export WANDB_MODE="offline"
export WANDB_RUN_NAME="poet-test"
# export CUDA_VISIBLE_DEVICES=$1
# export NUMEXPR_NUM_THREADS=64

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1


litgpt pretrain_npoet --config config_hub/pretrain/npoet_dev.yaml

exit 0

litgpt pretrain_poet Qwen3-8B \
    --data TinyLlama \
    --devices 1 \
    --num_nodes 1 \
    --seed 42 \
    --train.global_batch_size 16 \
    --train.micro_batch_size 1 \
    --train.log_interval 100 \
    --train.max_tokens 100000000 \
    --eval.interval 10000 \
    --logger_name wandb \
    --oft_block_size 128 \
    --oft_query True \
    --oft_key True \
    --oft_value True \
    --oft_head True\
    --oft_projection True \
    --oft_mlp True \
    --train.max_seq_length 2048 \
    --precision bf16-true \
    --poet.poet_update_reset_R_gap 200 \
    --poet.poet_lr 0.001 \

