from omegaconf import OmegaConf
from fairseq.criterions.masked_lm import MaskedLmConfig
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.masked_lm import MaskedLMTask

from .model import RobertaModel


def main() -> None:
    model_config = OmegaConf.create({'_name': 'roberta_small', 'max_positions': 512, 'dropout': 0.1, 'attention_dropout': 0.1})
    dictionary = Dictionary()
    for s in list('abcde'):
        dictionary.add_symbol(s)
    mlm_task = MaskedLMTask(MaskedLmConfig(), dictionary=dictionary)

    model = RobertaModel.build_model(model_config, task=mlm_task)

    print(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
    )


if __name__ == "__main__":
    main()
