

### [Cross domain generalization of autoformer](./AutoFormer)

AutoFormer is new one-shot architecture search framework dedicated to vision transformer search. It entangles the weights of different vision transformer blocks in the same layers during supernet training. 
Benefiting from the strategy, the trained supernet allows thousands of subnets to be very well-trained. Specifically, the performance of these subnets with weights inherited from the supernet is comparable to those retrained from scratch.
<div align="center">
    <img width="70%" alt="AutoFormer overview" src="AutoFormer/.figure/overview.png"/>
</div>

Through this project, we are evaluating the cross-domain generalization performace of Autoformer.

# How to run?

## Autoformer classification

### Run supernet train

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --data-path ./data/cifar-100 --data-set CIFAR100 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/classification/AutoFormer-T.yaml --epochs 500 --output ./output_cifar_scratchtrain
```

### Run retrain (From scratch)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --data-path ./data/cifar-100 --data-set CIFAR100 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/classification/AutoFormer-T.yaml --epochs 500 --output ./output_cifar_scratchretrain
```


### Run retrain (Finetune)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.classification.supernet_train --data-path ./data/cifar-100 --data-set CIFAR100 --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/classification/AutoFormer-T.yaml --resume /work/dlclarge1/sharmaa-dltrans/AutoFormer/AutoFormer/output_change/checkpoint.pth --start_epoch 500 --epochs 540 --output ./output_cifar_finetrain

```

## Autoformer for image super-resolution (SwinIR)

### Run supernet train

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode super --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/AutoFormer-T.yaml  --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json --epochs 300 --output ./output_swinir_super

```

### Run retrain (From scratch)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/AutoFormer-T.yaml --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json  --start_epoch 300 --output ./output_swinir_retrain_scratch

```

### Run retrain (Finetune)

```python
python3 -m torch.distributed.launch --use_env -m AutoFormer.experiments.super_resolution.supernet_train  --gp --change_qk --relative_position --mode retrain --dist-eval --cfg ./AutoFormer/experiments_configs/supernet-swinir/AutoFormer-T.yaml --resume ./AutoFormer/output_swin_newsplit/checkpoint.pth --opt-doc ./AutoFormer/experiments_configs/supernet-swinir/train_swinir_sr_lightweight.json  --start_epoch 300 --epochs 340 --output ./output_swinir_finetrain

```

## Autoformer for Roberta (maskedlm)

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

### Run retrain (Finetune)

```sh
bash 03-eval-test.sh
```