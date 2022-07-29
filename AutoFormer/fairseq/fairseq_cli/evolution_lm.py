import random
import sys
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path
from fairseq import models
from omegaconf import OmegaConf
#from fairseq_cli.search_validate import Search_Validate
#from AutoFormer.lib import utils
import argparse
import os
from fairseq.criterions.masked_lm import MaskedLmConfig
from fairseq.data.dictionary import Dictionary
from fairseq.tasks.masked_lm import MaskedLMTask
import yaml
from loguru import logger
#from AutoFormer.lib.config import cfg, update_config_from_file
import json

logger.add(sys.stdout, level='DEBUG')


def decode_cand_tuple(cand_tuple):
    logger.debug(f'cand tuple:{cand_tuple}')

    # Example: (256,256,256,256, 1024,1024,1024,1024,8,4,8,4, 4)
    depth = cand_tuple[-1]
    return list(cand_tuple[0:depth]), \
           list(cand_tuple[depth: 2 * depth]), \
           list(cand_tuple[2 * depth: 3 * depth]), \
           cand_tuple[-1]


# print (decode_cand_tuple(tuple((256,256,256,256, 1024,1024,1024,1024,8,4,8,4, 4))))


class EvolutionSearcher(object):

    def __init__(self, args, model, validator, choices, output_dir):
        #self.device = device
        self.model = model
        self.validator = validator
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits
        #self.val_loader = val_loader
        self.output_dir = output_dir
        self.s_prob = args.s_prob
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
        #logger.debug(f'vis dict:{self.vis_dict}')
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
            #logger.debug(f'random cand generated from mutation:{cands}')
            #logger.debug(f'vis dict:{self.vis_dict}')
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
        for dimension in dimensions:
            for i in range(depth):
                cand_tuple.append(random.choice(self.choices[dimension]))
        embed_dim = random.choice(self.choices['embed_dim'])
        for i in range(depth):
            cand_tuple.append(embed_dim)

        ffn_embed_dim = random.choice(self.choices['ffn_embed_dim'])
        for i in range(depth):
            cand_tuple.append(ffn_embed_dim)
        cand_tuple.append(depth)
        logger.debug(f'cand tuple:{cand_tuple}')
        return tuple(cand_tuple)

    def get_random(self, num):
        logger.debug('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
            logger.debug(f'random {len(self.candidates)}/{num}')
        logger.debug(f'random_num = {len(self.candidates)}')

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        logger.debug('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            num_heads, embed_dim, ff_embed_dim, depth= decode_cand_tuple(cand)
            random_s = random.random()

            # depth
            if random_s < s_prob:
                new_depth = random.choice(self.choices['depth'])

                if new_depth > depth:
                    num_heads = num_heads + [random.choice(self.choices['num_heads']) for _ in range(new_depth - depth)]
                    logger.debug(f'embed dim one:{embed_dim[-1]}, embed dim :{embed_dim}')
                    embed_dim = embed_dim + [embed_dim[-1] for _ in range(new_depth - depth)]
                    logger.debug(f'embed dim after:{embed_dim}')

                    ff_embed_dim = ff_embed_dim + [ff_embed_dim[-1] for _ in range(new_depth - depth)]
                    logger.debug(f'embed dim after:{ff_embed_dim}')
                else:
                    num_heads = num_heads[:depth]
                    embed_dim = embed_dim[:depth]
                    ff_embed_dim = ff_embed_dim[:depth]
                depth=new_depth

            # num_heads
            for i in range(depth):
                random_s = random.random()
                if random_s < m_prob:
                    num_heads[i] = random.choice(self.choices['num_heads'])

            # embed_dim
            random_s = random.random()
            if random_s < s_prob:
                logger.debug(f'sampled embed dim:{embed_dim}, ff_embed_dim:{ff_embed_dim}')
                sampled_embed_dim = random.choice(self.choices['embed_dim'])
                sampled_ff_embed_dim = random.choice(self.choices['ffn_embed_dim'])
                logger.debug(f'sampled config dim:{sampled_embed_dim}, sampled_ff_embed_dim:{sampled_ff_embed_dim}')
                for i in range(depth):
                    embed_dim[i] = sampled_embed_dim
                    ff_embed_dim[i] = sampled_ff_embed_dim
                    logger.debug(f'embed dim while sampling:{embed_dim}, ffn embed dim:{ff_embed_dim}')

            result_cand = num_heads + embed_dim + ff_embed_dim + [depth]

            logger.debug(f'mutated cand:{result_cand}')
            return tuple(result_cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.debug(f'mutation {len(res)}/{mutation_num}')

        logger.debug(f'mutation_num = {len(res)}')
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        logger.debug('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            logger.debug(f"cand p1:{p1}")
            logger.debug(f"cand p2:{p2}")
            crossover_cfg = []
            depth = p1[-1]

            #embed dim and ffn_embed dim has to be same for every layer, crossover just once and propogate the same
            embed_dim_index = list(range(0, 2 * depth))
            logger.debug(f'embed dim index:{embed_dim_index}')
            for idx, (i, j) in enumerate(zip(p1, p2)):
                if idx not in embed_dim_index:
                    crossover_cfg.append(random.choice([i, j]))
                else:
                    logger.debug(f'length of crossover cfg:{len(crossover_cfg)}, index:{idx + 1}')
                    if len(crossover_cfg) > idx:
                        logger.debug('index bigger than len of sampled cfg')
                        continue
                    embed_dim = random.choice([i, j])
                    logger.debug(f'sampled embed_dim:{embed_dim}')
                    for embed_per_block in range(len(embed_dim_index)):
                        crossover_cfg.append(embed_dim)
                        logger.debug(f'appending same embed dim to cfg list:{crossover_cfg}')
            logger.debug(f'final cfg list:{tuple(crossover_cfg)}')
            return tuple(crossover_cfg)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            logger.debug(f'crossover {len(res)}/{crossover_num}')

        print('crossover_num = {len(res)}')
        return res

    def search(self):
        print("inside evolutionary search")
        logger.debug(f'population_num = {self.population_num}'
                     f' select_num = {self.select_num}'
                     f' mutation_num = {self.mutation_num}'
                     f' crossover_num = {self.crossover_num}'
                     f' random_num = {self.population_num - self.mutation_num - self.crossover_num}'
                     f' max_epochs = {self.max_epochs}')

        # self.load_checkpoint()

        self.get_random(self.population_num)

        while self.epoch < self.max_epochs:
            logger.debug('epoch = {self.epoch}')

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['ppl'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['ppl'])

            logger.debug(f'epoch = {self.epoch} : top {len(self.keep_top_k[50])} result')
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[50]):
                logger.debug(f'No.{i + 1} {cand} val ppl = {self.vis_dict[cand]["ppl"]},'
                             f' params = {self.vis_dict[cand]["params"]}')
                tmp_accuracy.append(self.vis_dict[cand]['ppl'])
                with open('roberta_search.json', 'a+')as f:
                    json.dump({'epoch': self.epoch, 'rank': i + 1, 'ppl': self.vis_dict[cand]['ppl'],
                               'params': self.vis_dict[cand]['params'], 'cand': cand}, f)

                    f.write("\n")
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

            self.save_checkpoint()


def get_args_parser():
    parser = argparse.ArgumentParser('search strategy for roberta', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)

    # evolution search parameters
    parser.add_argument('--max-epochs', type=int, default=5)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    parser.add_argument('--param-limits', type=float, default=23)
    parser.add_argument('--min-param-limits', type=float, default=18)

    # config file
    #parser.add_argument('--config-dir',default='.AutoFormer/AutoFormer/fairseq/examples/roberta/config/pretraining',
    #                    help='experiment configure file name', required=True, type=str)

    # custom parameters
    parser.add_argument('--platform', default='pai', type=str, choices=['itp', 'pai', 'aml'],
                        help='Name of model to train')
    parser.add_argument('--teacher_model', default='', type=str,
                        help='Name of teacher model to train')
    parser.add_argument('--relative_position', action='store_true')
    parser.add_argument('--max_relative_position', type=int, default=14,
                        help='max distance in relative position embedding')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--gp', action='store_true')
    parser.add_argument('--change_qkv', action='store_true')

    # Model parameters
    parser.add_argument('--model', default='', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    # parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # custom model argument
    parser.add_argument('--rpe_type', type=str, default='bias', choices=['bias', 'direct'])
    parser.add_argument('--post_norm', action='store_true')
    parser.add_argument('--no_abs_pos', action='store_true')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr-power', type=float, default=1.0,
                        help='power of the polynomial lr scheduler')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01_101/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19', 'EVO_IMNET'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--no-amp', action='store_false', dest='amp')
    parser.set_defaults(amp=True)

    return parser



def main(args,model, search_validate, choices, output_dir='/work/dlclarge1/sharmaa-dltrans/robertasearch'):

    # parser = argparse.ArgumentParser('AutoFormer evolution search', parents=[get_args_parser()])
    # args = parser.parse_args()
    # if output_dir is not None:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # print(args)

    # seed = args.seed #+ utils.get_rank()
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(args.seed)
    # cudnn.benchmark = True


    # save config for later experiments_configs
    # with open(os.path.join(args.output_dir, "config.yaml"), 'w') as f:
    #     f.write(args_text)
    #
    # logger.debug(f"Creating LM model")
    # logger.debug(cfg)
    #
    #
    # validator = Search_Validate(cfg)
    # model = validator.model
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.debug(f'number of params:{n_parameters}')
    # if args.resume:
    #     if args.resume.startswith('https'):
    #         checkpoint = torch.hub.load_state_dict_from_url(
    #             args.resume, map_location='cpu', check_hash=True)
    #     else:
    #         checkpoint = torch.load(args.resume, map_location='cpu')
    #     logger.debug("resume from checkpoint: {args.resume}")

    # To Test


    t = time.time()
    print("call evolutionary search")
    searcher = EvolutionSearcher(args, model, search_validate, choices, output_dir)

    searcher.search()

    logger.debug('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))


