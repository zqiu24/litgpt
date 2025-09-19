# source .venv/bin/activate

# Resolve repo root and source wandb_api.sh from there
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"

[ -f "$REPO_ROOT/wandb_api.sh" ] && source "$REPO_ROOT/wandb_api.sh"
: "${WANDB_API_KEY:?WANDB_API_KEY not set; expected in repo-root/wandb_api.sh}"

# export WANDB_PROJECT="litgpt-test"
# export WANDB_MODE="offline"

litgpt pretrain_poet --config config_hub/pretrain/poet_dev_3b.yaml # --devices 1 --train.micro_batch_size 1
