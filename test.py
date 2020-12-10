import os
import numpy as np
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from load_data import PennFudanDataset

import utils

from pathlib import Path

from itertools import groupby
from pycocotools.coco import COCO
from pycocotools import mask as maskutil
# from detectron2.utils.visualizer import ColorMode

import matplotlib.pyplot as plt

import json

import pycocotools._mask as _mask

class Predictor():
    def __init__(self, model_name):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        
        self.num_classes = 21

        self.model = None
        self.get_model_instance_segmentation(self.num_classes)
        self.model_name = model_name

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
            ]),
        }

    def get_model_instance_segmentation(self, num_classes):
        # load an instance segmentation model
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=num_classes, pretrained_backbone=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    # return model

    def binary_mask_to_rle(self, binary_mask):
        rle = {'counts': [], 'size': list(binary_mask.shape)}
        counts = rle.get('counts')
        for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
        compressed_rle = maskutil.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
        compressed_rle['counts'] = str(compressed_rle['counts'], encoding='utf-8')
        return compressed_rle
    
    def encode(self, bimask):
        rles = []
        for instance_mask in bimask:
            np_mask = np.array(instance_mask[:,:,None], order='F', dtype='uint8')
            rle = maskutil.encode(np_mask)[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            rles.append(rle)
        return rles

    def predict(self):
        self.model.load_state_dict(torch.load(self.model_name))
        self.model.to(self.device)

        with torch.no_grad():
            self.model.eval()

            dataset = PennFudanDataset('dataset', train=False)

            testloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

            image_idxs = dataset.img_idxs
            

            results = []
            # results.append(image_names)
            
            k = 0
            for i in range(len(image_idxs)):
                imgId = image_idxs[i]
                print(imgId)

                img_info = dataset.coco.loadImgs(ids=imgId)[0]
                img_path = './dataset/test_images/'+img_info['file_name']
                img = Image.open(img_path).convert("RGB")
                # plt.imshow(img)
                img = self.data_transforms['test'](img)
                img = img.unsqueeze(0)
                inputs = img.to(self.device)
                outputs = self.model(inputs)[0]

                for i_instance in range(len(outputs['scores'])):
                    pred = {}
                    pred['image_id'] = imgId
                    pred['category_id'] = int(outputs['labels'][i_instance]) #shouldn't plus one
                    binary_mask = outputs['masks'][i_instance].to("cpu").squeeze().numpy()
                    
                    # convert float to binary
                    for bi in range(len(binary_mask)):
                        binary_mask[bi] = np.ndarray.round(binary_mask[bi])

                    pred['segmentation'] = self.binary_mask_to_rle(binary_mask)
                    # pred['segmentation'] = self.encode(binary_mask)[0]
                    pred['score'] = float(outputs['scores'][i_instance])
                    results.append(pred)

            with open('results.json', "w") as f:
                json.dump(results, f)
            # print(results)


def main():
    predictor = Predictor('./models/40.pt')
    predictor.predict()

if __name__ == '__main__':
    main()
