

if [ -z "$1" ]; then
  echo "Usage: $0 {0|1|2|3}"
  echo "  0: prepare_starcoder"
  echo "  1: prepare_slimpajama validation"
  echo "  2: prepare_slimpajama test"
  echo "  3: prepare_slimpajama train"
  exit 1
fi

case "$1" in
  0)
    python litgpt/data/prepare_starcoder.py \
      --input_dir data/starcoderdata-raw \
      --output_dir data/starcoderdata-llama3-8b \
      --tokenizer_path checkpoints/meta-llama/Llama-3.1-8B
    ;;
  1)
    python litgpt/data/prepare_slimpajama.py \
      --input_dir data/slimpajama-raw/validation \
      --output_dir data/slimpajama-llama3-8b/val \
      --tokenizer_path checkpoints/meta-llama/Llama-3.1-8B
    ;;
  2)
    python litgpt/data/prepare_slimpajama.py \
      --input_dir data/slimpajama-raw/test \
      --output_dir data/slimpajama-llama3-8b/test \
      --tokenizer_path checkpoints/meta-llama/Llama-3.1-8B
    ;;
  3)
    python litgpt/data/prepare_slimpajama.py \
      --input_dir data/slimpajama-raw/train \
      --output_dir data/slimpajama-llama3-8b/train \
      --tokenizer_path checkpoints/meta-llama/Llama-3.1-8B
    ;;
  *)
    echo "Invalid arg: $1. Use 0,1,2,3"
    exit 1
    ;;
esac