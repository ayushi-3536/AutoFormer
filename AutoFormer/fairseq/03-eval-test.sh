# Change the model path as needed
fairseq-validate "$(pwd)/data-bin/wikitext-103" --task "masked_lm" \
--path "$(pwd)/multirun/2022-07-25/00-33-55/0/checkpoints/checkpoint_last.pt" \
--valid-subset test --batch-size 32 --skip-invalid-size-inputs-valid-test