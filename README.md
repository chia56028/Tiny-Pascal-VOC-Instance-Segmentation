Tiny Pascal VOC Instance Segmentation
===
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

### Testing
```
python test.py
```

### Reference
- [TorchVision Object Detection Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)