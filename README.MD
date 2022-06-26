# ECCV'22 ChaLearn Seasons in Drift Challenge (track 2: Week level)


## 1. Enviroment

+ Ubuntu 18.04
+ CUDA 11.1 
+ 8*3090 GPU
+ Python3.7
+ pytorch-1.7, torchvision-0.8.0
+ mmdetection-2.22.0

## 2. Install
### (1). Create a conda environment and activate it.
```shell
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

### (2). Install PyTorch 1.7 or highre

```shell
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
```
### (3). Install MMdetection-2.22.0 and mmcv

```shell
pip install -r requirements.txt
pip install mmdet==2.22.0
pip install mmcv-full==1.3.17
pip install tqdm
```

# 3. data pross

Notice: We only use 30% labels to train our model.

（1）Place the training set in the `data/train` folder and unzip it.

（2）Place the testing set in the `data/test` folder and unzip it.

（3）Place the labels file in the `data/labels` folder and unzip it.

（4) Run dataprocess: `sh data_process.sh`

# 4. training models
Notice: We use 8*3090 GPU to train our models. If GPU numbers not equal8, we should simply change the learning rate.
For examole: If 4 GPUs are used, change the learning rate 0.00001/2 in `config_week/week.py` line 404.

Before training, pre-trained weights on coco dataset should download unzip, and put it in `weights` folder

```shell
wget https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip

mv https://github.com/CBNetwork/storage/releases/download/v1.0.0/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip weights/

unzip cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth.zip

sh dist_train.sh
```

# 5. test and generate result file

```shell
sh test_submit.sh
```
Now we can find a zip file in `submit_last` folder.

### If you have any problems, pleanse mail me directly.
### Thanks for your efforts.