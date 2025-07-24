# #!/bin/bash
GOVREPORT="--dataset emozilla/govreport-test-tokenized --tokenized --dataset-min-tokens 16384 --samples 50"


# Example script
# Calculate LongPPL using an online evaluator model
python perplexity.py \
    ${GOVREPORT} \
    --model Qwen/Qwen2-7B \
    --evaluator-model meta-llama/Llama-3.1-8B \
    --mode online \
    --alpha 2.0 \
    --beta -2.0

# Calculate LongPPL using offline computed key tokens (default available: [Qwen2-72B-Instruct, Mistral-Large-Instruct-2407, Meta-Llama-3.1-8B])
python perplexity.py \
    ${GOVREPORT} \
    --model Qwen/Qwen2-7B \
    --evaluator-name Meta-Llama-3.1-8B \
    --mode offline