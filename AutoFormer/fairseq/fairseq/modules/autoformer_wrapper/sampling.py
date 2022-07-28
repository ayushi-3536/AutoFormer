import random

from typing import Dict


def sample_config(choices: Dict):

    config = {}
    dimensions = ['FFN_EMBED_DIM', 'NUM_HEADS']
    config['DEPTH'] = random.choice(choices['DEPTH'])
    for dimension in dimensions:
        config[dimension] = [random.choice(choices[dimension]) for _ in range(config['DEPTH'])]

    config['EMBED_DIM'] = [random.choice(choices['EMBED_DIM'])] * config['DEPTH']

    return config