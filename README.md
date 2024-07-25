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
follow the README to download the checkpoint to files correspondingly.


```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e .  # also optionally follow fairseq README for apex installation for fp16 training.
export MKL_THREADING_LAYER=GNU  # fairseq may need this for numpy.

cd examples/MMPT  # MMPT can be in any folder, not necessarily under fairseq/examples.
pip install -e .

```
### Download Checkpoints

VideoCLIP use pre-trained [S3D](https://github.com/antoine77340/S3D_HowTo100M) for video feature extraction. Please place the models as `pretrained_models/s3d_dict.npy` and `pretrained_models/s3d_howto100m.pth`.

Download VideoCLIP checkpoint `https://dl.fbaipublicfiles.com/MMPT/retri/videoclip/checkpoint_best.pt` to `runs/retri/videoclip` .

#### Demo of Inference
run `python locallaunch.py projects/retri/videoclip.yaml --dryrun` to get all `.yaml`s for VideoCLIP.


put the scripts `process_coin.py`, `process_crosstask.py`, `process_niv.py` under `cd /fairseq/examples/MMPT`
change the `save_root_path` and the times of data augmentation `aug_times`.

```
cd /fairseq/examples/MMPT

python process_coin.py
```
run the script to get the extracted features and save them to .npy files. 

## Train
### Dataloder
the corresponding dataloader is `data_load_action_classifier` 

### Set arguments 
Set arguments in `train_action_classifer.sh`. Train task prediction model for each dataset. Set `--class_dim, --action_dim, --observation_dim` accordingly.
 To train the model, run

```
sh train_action_classifer.sh
```

