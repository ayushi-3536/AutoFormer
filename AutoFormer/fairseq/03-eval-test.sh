# Change the model path as needed
fairseq-validate "$(pwd)/data-bin/wikitext-103" --task "masked_lm" \
--path "$(pwd)/multirun/2022-08-01/00-21-42/0/retrain/checkpoint_best.pt" \
--valid-subset test --batch-size 32 --skip-invalid-size-inputs-valid-test
