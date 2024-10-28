# #!/bin/bash
GOVREPORT="--tokenized emozilla/govreport-test-tokenized --dataset-min-tokens 16384 --samples 50"

python perplexity.py \
    ${GOVREPORT} \
    --model $evaluated_model \
    --evaluator-name $evaluator_name \
    --evaluator-model $evaluator_model \
    --mode offline

