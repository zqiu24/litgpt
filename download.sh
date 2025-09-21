huggingface-cli download cerebras/slimpajama-627b \
  --repo-type dataset \
  --include "validation/**" \
  --local-dir /home/fdraye/litgpt/data/slimpajama-raw

exit 0

huggingface-cli download cerebras/SlimPajama-627B "validation/*" \
  --repo-type dataset --local-dir ./SlimPajama-validation

git clone --filter=blob:none https://huggingface.co/datasets/cerebras/slimpajama-627b \
   /home/fdraye/litgpt/data/slimpajama-raw
cd  /home/fdraye/litgpt/data/slimpajama-raw
git sparse-checkout init --cone
git sparse-checkout set validation
git lfs fetch --include="validation/**" --exclude=""
git lfs checkout


git clone https://huggingface.co/datasets/cerebras/slimpajama-627b data/slimpajama-raw
git clone https://huggingface.co/datasets/bigcode/starcoderdata data/starcoderdata-raw