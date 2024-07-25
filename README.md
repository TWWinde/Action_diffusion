# How to extract features and train an action classifier



## Dataset
We will use 3 datasets:

### Crosstask

```
cd dataset/crosstask
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://vision.eecs.yorku.ca/WebShare/CrossTask_s3d.zip
unzip '*.zip'
```
located in 
/data/crosstask_release

### Coin

```
cd dataset/coin
wget https://vision.eecs.yorku.ca/WebShare/COIN_s3d.zip
unzip COIN_s3d.zip
```
located in 
/data/COIN/


### NIV

```
cd dataset/NIV
wget https://vision.eecs.yorku.ca/WebShare/NIV_s3d.zip
unzip NIV_s3d.zip
```
located in 
/data/niv/

## Features Extraction

We implement [VideoCLIP](https://github.com/facebookresearch/fairseq/tree/main/examples/MMPT#readme) model to extract feature

```
git clone https://github.com/facebookresearch/fairseq.git

cd /fairseq/examples/MMPT
```
put the scripts `process_coin`, `process_crosstask`, `process_niv` under `cd /fairseq/examples/MMPT`



## Train

### Task Predicion

Set arguments in `train_mlp.sh`. Train task prediction model for each dataset. Set `--class_dim, --action_dim, --observation_dim` accordingly.  For horizon `T={3,4,5,6}`, set `--horizon, --json_path_val ,--json_path_train` accordingly.

```
sh train_mlp.sh
```

Set the checkpoint path in `temp.py` via `--checkpoint_mlp`


### Diffusion Model

Set `dataset, horizon` in `train.sh` to corresponding datasets and time horizons for training. Set `mask_type` to `multi_add` to use multiple-add noise mask or `single_add` to use single-add noise mask. Set `attn` to `WithAttention` to use UNet with attention or `NoAttention` to use UNet without attention.

To train the model, run

```
sh train.sh
```

## Inference

Set `dataset, horizon` in `inference.sh` to corresponding datasets and time horizons for training. Set `checkpoint_diff` to the pre-trained model.
Set `mask_type` to `multi_add` to use multiple-add noise mask or `single_add` to use single-add noise mask. Set `attn` to `WithAttention` to use UNet with attention or `NoAttention` to use UNet without attention.
