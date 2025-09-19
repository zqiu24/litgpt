# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

# export WANDB_PROJECT="sPOET"
# export WANDB_RUN_NAME="adamw-1b"
# export CUDA_VISIBLE_DEVICES=0
# export NUMEXPR_NUM_THREADS=64
# export WANDB_MODE="offline"

# Install dependencies
# set num_workers to 32 in litgpt/litgpt/data/tinystories.py
# lightning==2.5.0.post0


litgpt pretrain --config config_hub/pretrain/adamw_baseline.yaml

exit 0


litgpt pretrain Qwen3-8B \
    --data TinyLlama \
    --logger_name wandb \
    --devices 1 \
    --num_nodes 1 \
    --seed 42 \
    --train.global_batch_size 16 \
    --train.micro_batch_size 1 \
    --train.max_seq_length 2048 \
    --train.log_interval 100 \
    --train.max_tokens 100000000 \
    --precision bf16-true \

exit 0


USE_TORCH_COMPILE=false litgpt pretrain --config config_hub/pretrain/adamw_baseline.yaml

exit 0

litgpt pretrain \
    --config config_hub/pretrain/tinyllama.yaml \
    --data TinyStories \
    --tokenizer_dir hf_models/meta-llama/Llama-2-7b-hf \
    --logger_name wandb \
    --devices auto \
    --num_nodes $num_nodes \
    --devices 8 \
    --seed 42

node_rank=$1
num_nodes=1

litgpt download NousResearch/Hermes-2-Pro-Mistral-7B \
 --model_name Mistral-7B-v0.1


fabric run \
    --node-rank=$node_rank  \
    --main-address=172.22.8.7 \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=$num_nodes \
    litgpt pretrain \
    --config config_hub/pretrain/tinyllama.yaml \
    --data TinyStories \
    --tokenizer_dir hf_models/meta-llama/Llama-2-7b-hf \
    --logger_name wandb \
    --devices auto \
    --num_nodes $num_nodes \
    --devices 8 \
    --seed 42


exit 0


torchrun \
    --accelerator=cuda \
    --devices=8 \
    --num-nodes=$num_nodes \
    --node-rank=$node_rank \
    litgpt.pretrain.py \
    --config config_hub/pretrain/tinyllama.yaml \
    --data TinyStories \
    --tokenizer_dir hf_models/meta-llama/Llama-2-7b-hf \
    --logger_name wandb \
    --devices auto \
    --num_nodes $num_nodes \
    --seed 42
