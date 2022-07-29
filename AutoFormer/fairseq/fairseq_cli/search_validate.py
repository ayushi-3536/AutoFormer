#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
from argparse import Namespace
from itertools import chain
import time
import torch
from omegaconf import DictConfig

from fairseq_cli.evolution_lm import EvolutionSearcher, main as evolution_init
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.search_validate")

class Search_Validate:

    def __init__(self, cfg: DictConfig, override_args=None):
        if isinstance(cfg, Namespace):
            cfg = convert_namespace_to_omegaconf(cfg)
        utils.import_user_module(cfg.common)
        self.cfg=cfg
        print("cfg in main",self.cfg)
        utils.import_user_module(cfg.common)
        print("common eval in cfg",cfg.common_eval)
        reset_logging()

        assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
        ), "Must specify batch size either with --max-tokens or --batch-size"

        use_fp16 = cfg.common.fp16
        self.use_cuda = torch.cuda.is_available() and not cfg.common.cpu

        if self.use_cuda:
            torch.cuda.set_device(cfg.distributed_training.device_id)

        if cfg.distributed_training.distributed_world_size > 1:
            self.data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
            self.data_parallel_rank = distributed_utils.get_data_parallel_rank()
        else:
            self.data_parallel_world_size = 1
            self.data_parallel_rank = 0

        if override_args is not None:
            overrides = vars(override_args)
            overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
        else:
            overrides = None

        # Load ensemble
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        models, self.saved_cfg, self.task = checkpoint_utils.load_model_ensemble_and_task(
            [cfg.common_eval.path],
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
        )
        self.model = models[0]
        print("model",self.model)
        print("save config",self.saved_cfg)
        print("self.task",self.task)
        # Print args
        #logger.info(self.saved_cfg)

        # Build criterion
        self.criterion = self.task.build_criterion(self.saved_cfg.criterion)
        self.criterion.eval()

    def evaluate(self, config):
        print("current device",torch.cuda.current_device())
        self.model.to(torch.cuda.current_device()) 
        self.model.set_sample_config(sample_embed_dim=config['embed_dim'],
            sample_ffn_embed_dim=config['ffn_embed_dim'],
            sample_num_heads=config['num_heads'],
            sample_depth=config['depth'],
        )
        print("Done setting model with subnet")

        for subset in self.cfg.dataset.valid_subset.split(","):
            print('subset',subset)
            try:
                self.task.load_dataset(subset, combine=False, epoch=1, task_cfg=self.saved_cfg.task)
                dataset = self.task.dataset(subset)
            except KeyError:
                raise Exception("Cannot find dataset: " + subset)

            # Initialize data iterator
            itr = self.task.get_batch_iterator(
                dataset=dataset,
                max_tokens=self.cfg.dataset.max_tokens,
                max_sentences=self.cfg.dataset.batch_size,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    *[self.model.max_positions()],
                ),
                ignore_invalid_inputs=self.cfg.dataset.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.cfg.dataset.required_batch_size_multiple,
                seed=self.cfg.common.seed,
                num_shards=self.data_parallel_world_size,
                shard_id=self.data_parallel_rank,
                num_workers=self.cfg.dataset.num_workers,
                data_buffer_size=self.cfg.dataset.data_buffer_size,
            ).next_epoch_itr(shuffle=False)
            print("itr",itr)
            progress = progress_bar.progress_bar(
                itr,
                log_format=self.cfg.common.log_format,
                log_interval=self.cfg.common.log_interval,
                prefix=f"valid on '{subset}' subset",
                default_log_format=("tqdm" if not self.cfg.common.no_progress_bar else "simple"),
            )

            log_outputs = []
            print("start eval")
            for i, sample in enumerate(progress):
                sample = utils.move_to_cuda(sample) if self.use_cuda else sample
                _loss, _sample_size, log_output = self.task.valid_step(sample, self.model, self.criterion)
                print("log output",log_output)
                progress.log(log_output, step=i)
                log_outputs.append(log_output)


            if self.data_parallel_world_size > 1:
                log_outputs = distributed_utils.all_gather_list(
                    log_outputs,
                    max_size=self.cfg.common.all_gather_list_size,
                    group=distributed_utils.get_data_parallel_group(),
                )
                log_outputs = list(chain.from_iterable(log_outputs))

            with metrics.aggregate() as agg:
                self.task.reduce_metrics(log_outputs, self.criterion)
                log_output = agg.get_smoothed_values()

            progress.print(log_output, tag=subset, step=i)
            print('log output',log_output)
            return log_output


#Todo:see if needed to extract config

def cli_main():
    parser = options.get_search_validation_parser()
    args = options.parse_args_and_arch(parser)
    print("before parsing",args)
    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)
    print("after override",override_args)

    search_validate = Search_Validate(convert_namespace_to_omegaconf(override_args))
    model = search_validate.model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    choices = {'embed_dim': [256, 384, 512],
               'ffn_embed_dim': [1024, 1536, 2048],
               'num_heads': [4, 8],
               'depth': [4, 5, 6]
               }

    t = time.time()

    print('start searching')    
    searcher = evolution_init(args,
                              model=model,
                              search_validate=search_validate,
                              choices=choices,
                              output_dir='/work/dlclarge1/sharmaa-dltrans/robertasearch')

    #searcher.search()

    logger.debug('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))




