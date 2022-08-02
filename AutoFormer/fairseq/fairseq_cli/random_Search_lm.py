import random
import sys
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from fairseq import models
from omegaconf import OmegaConf
# from fairseq_cli.search_validate import Search_Validate
# from AutoFormer.lib import utils
import argparse
import os
from fairseq.criterions.masked_lm import MaskedLmConfig
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.masked_lm import MaskedLMTask
import yaml
from loguru import logger
# from AutoFormer.lib.config import cfg, update_config_from_file
import json

logger.add(sys.stdout, level='DEBUG')


def decode_cand_tuple(cand_tuple):
    # logger.debug(f'cand tuple:{cand_tuple}')

    # Example: (256,256,256,256, 1024,1024,1024,1024,8,4,8,4, 4)
    depth = cand_tuple[-1]
    return list(cand_tuple[0:depth]), \
           list(cand_tuple[depth: 2 * depth]), \
           list(cand_tuple[2 * depth: 3 * depth]), \
           cand_tuple[-1]


# print (decode_cand_tuple(tuple((256,256,256,256, 1024,1024,1024,1024,8,4,8,4, 4))))


class RandomSearcher(object):

    def __init__(self, args, model, validator, choices, output_dir):
        # self.device = device
        self.model = model
        self.validator = validator
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = 50
        self.population_num = 50
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        # self.val_loader = val_loader
        self.output_dir = output_dir
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.checkpoint_path = args.resume
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.choices = choices

    def save_checkpoint(self):

        info = {}
        info['top_accuracies'] = self.top_accuracies
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        checkpoint_path = os.path.join(self.output_dir, "checkpoint-{}.pth.tar".format(self.epoch))
        torch.save(info, checkpoint_path)
        logger.debug(f'save checkpoint to:{checkpoint_path}')

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            return False
        info = torch.load(self.checkpoint_path)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        # logger.debug(f'vis dict:{self.vis_dict}')
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        logger.debug(f'load checkpoint from:{self.checkpoint_path}')
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            print('visited already')
            return False
        num_heads, embed_dim, ff_embed_dim, depth = decode_cand_tuple(cand)

        logger.debug(
            f'embed_dim:{embed_dim}, ffn_embed_dim"{ff_embed_dim},'
            f' num_heads:{num_heads}, depth:{depth}')
        sampled_config = {}
        sampled_config['depth'] = depth
        sampled_config['num_heads'] = num_heads
        sampled_config['embed_dim'] = embed_dim
        sampled_config['ffn_embed_dim'] = ff_embed_dim

        logger.debug(f'sampled_config:{sampled_config}')

        self.model.set_sample_config(sample_num_heads=num_heads,
                                     sample_embed_dim=embed_dim,
                                     sample_ffn_embed_dim=ff_embed_dim,
                                     sample_depth=depth)

        n_parameters = self.model.get_sampled_params_numel()
        info['params'] = n_parameters / 10. ** 6

        if info['params'] > self.parameters_limits:
            logger.debug('parameters limit exceed')
            return False

        if info['params'] < self.min_parameters_limits:
            logger.debug('under minimum parameters limit')
            return False

        logger.debug(f"cand:{cand},params:{info['params']}")
        eval_stats = self.validator.evaluate(config=sampled_config)
        # eval_stats = (self.val_loader, self.model, self.model.module, self.device, amp=self.args.amp, mode='retrain',
        #               retrain_config=sampled_config)
        logger.debug(f'eval stats:{eval_stats}')
        info['ppl'] = eval_stats['ppl']
        # info['test_acc'] = test_stats['acc1']

        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        logger.debug('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            # logger.debug(f'random cand generated from mutation:{cands}')
            # logger.debug(f'vis dict:{self.vis_dict}')
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        '''
        Generate random configuration:
        Example: 'embed_dim': [256,256,256,256], ffn_embed_dim: [1024,1024,1024,1024],'num_heads': [8,4,8,4], 'depth': 4
        #
        Returns:
            Tuple: Sampled configuration
        '''

        cand_tuple = list()
        dimensions = ['num_heads']

        depth = random.choice(self.choices['depth'])

        for i in range(depth):
            cand_tuple.append(random.choice(self.choices['num_heads']))

        embed_dim = random.choice(self.choices['embed_dim'])
        for i in range(depth):
            cand_tuple.append(embed_dim)

        for i in range(depth):
            cand_tuple.append(random.choice(self.choices['ffn_embed_dim']))

        cand_tuple.append(depth)
        logger.debug(f'cand tuple:{cand_tuple}')
        return tuple(cand_tuple)

    def get_random(self, num):
        logger.debug('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        print("len of can",len(self.candidates))
        print("pop num desired",num)

        while len(self.candidates) < num:
            print("l of c",len(self.candidates))
            cand = next(cand_iter)
            if not self.is_legal(cand):
                print("cand not leag")
                continue
            print("cand legal appending")
            self.candidates.append(cand)
            logger.debug(f'random {len(self.candidates)}/{num}')
        logger.debug(f'random_num = {len(self.candidates)}')

    def search(self):
        print("inside evolutionary search")
        logger.debug(f'population_num = {self.population_num}'
                     f' select_num = {self.select_num}'
                     f' max_epochs = {self.max_epochs}')

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            logger.debug('epoch = {self.epoch}')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['ppl'], reverse=False)
            # self.update_top_k(
            #     self.candidates, k=50, key=lambda x: self.vis_dict[x]['ppl'], reverse=False)

            logger.debug(f'epoch = {self.epoch} : top {len(self.keep_top_k[50])} result')
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                logger.debug(f'No.{i + 1} {cand} val ppl = {self.vis_dict[cand]["ppl"]},'
                             f' params = {self.vis_dict[cand]["params"]}')
                tmp_accuracy.append(self.vis_dict[cand]['ppl'])
                with open('roberta_search_random.json', 'a+')as f:
                    json.dump({'epoch': self.epoch, 'rank': i + 1, 'ppl': self.vis_dict[cand]['ppl'],
                               'params': self.vis_dict[cand]['params'], 'cand': cand}, f)

                    f.write("\n")
                    f.close()
            self.top_accuracies.append(tmp_accuracy)
            self.candidates=[]
            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()


def main(args, model, search_validate, choices, output_dir='/work/dlclarge1/sharmaa-dltrans/robertasearch_random'):
    t = time.time()
    print("call random search")
    searcher = RandomSearcher(args, model, search_validate, choices, output_dir)

    searcher.search()

    logger.debug('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))
