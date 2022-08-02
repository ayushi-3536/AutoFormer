

# [Cross Domain Generalization of AutoFormer](./AutoFormer)

## Background

AutoFormer [1] is an one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.
<div align="center">
    <img width="70%" alt="AutoFormer overview" src="AutoFormer/.figure/overview.png"/>
</div>

## Overview

In this work, we evaluate the cross-domain generalization performace of Autoformer.

Specifically we explore AutoFormer performance on:
- **Image Classification.** [1] does it for ImageNet. We experiment with CIFAR-100.
- **Image Super-Resolution.** We experiment with an AutoFormer version of SwinIR [2] trained on the DIV2K dataset and evaluated on Set14.
- **(Masked) Language Modeling.** We experiment with an AutoFormer version of RoBERTa [3] with WikiText-103.

## Setup

To set up the enviroment you can easily run the following command:

```buildoutcfg
conda create -n Autoformer python=3.6
conda activate Autoformer
pip install -r AutoFormer/requirements-dev.txt
```

Also, setup `fairseq`, whose RoBERTa implementation we have modified.

```buildoutcfg
cd AutFormer/AutoFormer/fairseq
pip install --editable ./
```

## Autoformer Classification

### Run supernet train

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --num_workers=8 --data-path ./data/cifar-100 --data-set CIFAR100 --gp  --change_qk --relative_position --mode super --dist-eval --cfg ./AutoFormer/experiments_configs/classification/supernet-T.yaml --epochs 500 --warmup-epochs 20 --output ./output_change --batch-size 128```
```

### Run evolutionary search
```python
python -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.evolution --data-path ./data/evol_cifar100 --gp  --change_qk --relative_position --dist-eval --cfg ./AutoFormer/experiments_configs/classification/supernet-T.yaml --resume {/PATH/checkpoint.pth}  --min-param-limits 0.0 --param-limits 5.560792 --data-set CIFAR100 --output_dir ./search_output
```

### Run retrain (From scratch)
```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --data-path ./data/cifar-100 --data-set CIFAR100 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/classification/AutoFormer-T.yaml --epochs 500 --output ./output_cifar_scratchretrain
```

### Run retrain (Finetune)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --data-path ./data/cifar-100 --data-set CIFAR100 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/classification/AutoFormer-T.yaml --resume {/PATH/checkpoint.pth} --start_epoch 500 --epochs 540 --output ./output_cifar_finetrain

```

## Autoformer for image super-resolution (SwinIR)

### Run supernet train

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode super --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/supernet-T.yaml --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json --epochs 300 --output ./output_swinir_train
```

### Run evolutionary search
```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.random_swin --gp  --change_qk --relative_position --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/supernet-T.yaml  --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json --resume {/PATH/checkpoint.pth}  --min-param-limits 0.0 --param-limits 9.29628  --output_dir ./search_swinir_random```
```

### Run retrain (From scratch)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/AutoFormer-T.yaml  --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json --epochs 300 --output ./output_swinir_retrain
```

### Run Finetune

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/AutoFormer-T.yaml --resume {/PATH/checkpoint.pth} --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json  --start_epoch 300 --epochs 340 --output ./output_swinir_finetrain
```

## Autoformer for RoBERTa (Masked-LM)

Note: Assumes you are inside `AutoFormer/AutoFormer/fairseq`.

### Setup data

```sh
bash 01-data-setup.sh
```

### Run supernet train

```sh
bash 02-train.sh
```
### Run retrain (From scratch)

```sh
bash 05-retrain-from-scratch.sh
```

### Run retrain (Finetune)

```sh
bash 06-retrain-finetune.sh
```

### Evaluate on test set

```sh
bash 03-eval-test.sh
```

## Model Checkpoints
- https://drive.google.com/drive/folders/1qX4hld4xOS9MKTj6Rzz7t_f0QP8Eqckv?usp=sharing
## Code Sources

- [microsoft/Cream](https://github.com/microsoft/Cream)
- [cszn/KAIR](https://github.com/cszn/KAIR)
- [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)

## References

[1] Chen, Minghao, et al. "Autoformer: Searching transformers for visual recognition." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[2] Liang, Jingyun, et al. "Swinir: Image restoration using swin transformer." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[3] Liu, Yinhan, et al. "Roberta: A robustly optimized bert pretraining approach." arXiv preprint arXiv:1907.11692 (2019).