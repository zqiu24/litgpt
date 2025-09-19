# . /home/zqiu/anaconda3/etc/profile.d/conda.sh
# module load cuda/12.9
# conda activate test

source .venv/bin/activate

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

# export WANDB_MODE="offline"
# export WANDB_RUN_NAME="poet-test"
# export CUDA_VISIBLE_DEVICES=$1
# export NUMEXPR_NUM_THREADS=64

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_SHOW_CPP_STACKTRACES=1


litgpt pretrain_lora --config config_hub/pretrain/lora_dev.yaml

