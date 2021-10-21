# unet.pytorch
U-Net Convolutional Neural Network for Brain MRI Segmentation dataset

![](https://github.com/gaungalif/unet.pytorch/blob/main/results/unet_brain_MRI_results.gif)

This repository contains the code to train Brain MRI Segmentation datasets taken from kaggle as an image segmentation using pytorch. you can download the weights that i have been trained for 800 epochs here, or you can try to use the onnx weight predictor version here 
- store the weight at `weights/`

## Requirment:
- onnx==1.10.1
- onnxruntime==1.9.0
- torch==1.8.2
- torchvision==0.9.2 
- torchaudio==0.8.2
- python==3.8.0
- pytorch-lightning==1.4.5
- opencv-python==4.4.0.46
- opencv-python-headless==4.5.1.48
- fire==0.4.0

## Hardware Requirment:
- Computer with decend RAM and CPU
- GPU (optional)

## How to Use:
### Dataset:
- Download the dataset manually from here: https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation/download
- Store dataset at `dataset/brain_mri/`

### Training:
- Use `train.py` to train the model.
- Change `dataset` path to the appropriate path if needed
- You can modify the Hyperparameter and Augmentation if needed
- Use this command 'python train.py --help' for help

example command: 
```
python train.py  --max_epoch 100 --batch_size 64 --num_workers 2
```

### Test:
- Use 'test.py' to test the model that you have trained.
- modify the data_dir(default: ./dataset/brain_mri/) and weight(default: weiights/*.pth) or weight_onnx(default: weights/*.pth.onnx)to the specific path location to test the image
- change the mode into "torch" or "onnx" and compare the differents
change the `--idx` to change the images 
- the program will show 3 images original images, predicted, and the labels

download the weights that i have been trained for 800 epochs here: https://drive.google.com/file/d/1RdfAtRuKtJBg6K99X7a_K4EQ_aDcOdT2/view?usp=sharing

or you can try to use the onnx weight predictor version here: 
https://drive.google.com/file/d/1RDygW-d8-o6EFpdx7v3j0HtJApJB6BLP/view?usp=sharing

example command: 
```
python test.py --image_path ./dataset/brain_mri/ --idx 23 --weight_path --mode onnx
```
- output :
![](https://github.com/gaungalif/unet.pytorch/blob/main/results/unet_brain_MRI_results.gif)

## Reference:

- Mateuszbuda, lgg-mri-segmentation , https://www.kaggle.com/mateuszbuda/lgg-mri-segmentation/download
- Nunenuh, UNet, Pytorch, https://github.com/nunenuh/unet.pytorch
- Torchvision Documentation, Pytorch, https://pytorch.org/vision/stable/index.html
- Pytorch Lightning Documentation, https://pytorch-lightning.readthedocs.io/en/latest/
- Pytorch ONNX, https://pytorch.org/docs/stable/onnx.html