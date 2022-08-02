from omegaconf import OmegaConf
from fairseq import models
from fairseq.criterions.masked_lm import MaskedLmConfig
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.masked_lm import MaskedLMTask

from .model import RobertaModel


def main() -> None:
    model_config = OmegaConf.create({'_name': 'roberta_large', 'max_positions': 512, 'dropout': 0.1, 'attention_dropout': 0.1})
    dictionary = Dictionary()

    # Vocabulary size for RoBERTa
    # It changes the embedding weight and thus the num of params in the model
    for i in range(50_000):
        dictionary.add_symbol(i)
    mlm_task = MaskedLMTask(MaskedLmConfig(), dictionary=dictionary)

    # model = RobertaModel.build_model(model_config, task=mlm_task)
    model = models.build_model(model_config, mlm_task)

    # This overestimates the model params because there are mutliple conditional pathways in the forward pass
    # So many params always remain unused
    print(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )

    print("Model params (RoBERTa Large): {:,}".format(model.get_sampled_params_numel()))

    # Setting config to be the same as roberta base
    # The num of params should thus be 125M
    sample_config = {
        'sample_embed_dim': [768] * 12,
        'sample_ffn_embed_dim': [3072] * 12,
        'sample_num_heads': [12] * 12,
        'sample_depth': 12
    }
    model.set_sample_config(**sample_config)
    print("Model sampled params (RoBERTa base): {:,}".format(model.get_sampled_params_numel()))


if __name__ == "__main__":
    main()
