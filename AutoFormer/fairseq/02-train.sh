fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name small task.data="$(pwd)/data-bin/wikitext-103"