fairseq-hydra-train -m --config-dir examples/roberta/config/pretraining \
--config-name small task.data="$(pwd)/data-bin/wikitext-103" common.search_config="$(pwd)/examples/roberta/config/pretraining/supernet-T.yaml" \
common.train_mode="retrain" checkpoint.restore_file="$(pwd)/checkpoint_best.pt" checkpoint.save_dir="retrain"
