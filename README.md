Tiny Pascal VOC Instance Segmentation
===
The proposed challenge is a tiny Pascal VOC instance segmentation.
Contains two parts:
1. Do segmentation mask for each instance
2. classify the instances into 20 classes
The giving tiny Pascal VOC dataset contains 1349 images for training and 100 images for testing. This project uses the Pytorch library and backbone pre-trained on ImageNet model to fix this challenge.

### Hardware
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA GeForce RTX 2080 Ti

### Environment
- Microsoft win10
- Python 3.7.3
- Pytorch 1.7.0
- CUDA 10.2

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#install-packages)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Testing](#testing)
5. [Reference](#reference)

### Install Packages
- install pytorch from https://pytorch.org/get-started/locally/
- install dependencies
```
pip install -r requirements.txt
```

### Data Preparation
Download the given dataset from [Google Drive](https://drive.google.com/drive/u/4/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK).
```
data /
  +- train_images
  +- test_images
  +- pascal_train.json
  +- test.json
```

### Training
```
python train.py
```

※ get more info by `python train.py --help`
```
usage: train.py [-h] [-r WORKING_DIR] [-e EPOCH] [-b BATCH_SIZE]
                [-lr LEARNING_RATE]

optional arguments:
  -h, --help            show this help message and exit
  -r WORKING_DIR, --root WORKING_DIR
                        path to dataset
  -e EPOCH, --epochs EPOCH
                        num of epoch
  -b BATCH_SIZE, --batch BATCH_SIZE
                        set batch size
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        set learning rate
```

### Testing
```
python test.py
```

### Reference
- [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)