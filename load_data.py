import os
import numpy as np
import torch
from PIL import Image

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torchvision.transforms as transforms

import transforms as T

import matplotlib.pyplot as plt

class PennFudanDataset(object):
    def __init__(self, root, train):
        self.root = root

        self.is_train = train
        if self.is_train==True:
            self.img_path = os.path.join(root, "train_images")
            self.json_path = os.path.join(root, "pascal_train.json")
        else:
            self.img_path = os.path.join(root, "test_images")
            self.json_path = os.path.join(root, "test.json")

        self.imgs = list(sorted(os.listdir(self.img_path)))

        self.coco = COCO(self.json_path)
        self.img_idxs = list(self.coco.imgs.keys())

        self.data_transforms = self.get_transform()

    def get_transform(self):
        trans = []
        trans.append(T.ToTensor())
        if self.is_train:
            trans.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(trans)


    def __getitem__(self, idx):
        idx = self.img_idxs[idx]
        
        coco = self.coco

        img_info = coco.loadImgs(ids=idx)[0]

        # load images
        img_path = os.path.join(self.img_path, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        # plt.imshow(img)

        target = {}
        if self.is_train==True:
            annids = coco.getAnnIds(imgIds=idx)
            anns = coco.loadAnns(annids)

            # count number of objects
            num_objs = len(annids)

            boxes = []
            labels = []
            masks = []
            iscrowds = []
            for i in range(num_objs):
                xmin = anns[i]['bbox'][0]
                xmax = anns[i]['bbox'][0]+anns[i]['bbox'][2]
                ymin = anns[i]['bbox'][1]
                ymax = anns[i]['bbox'][1]+anns[i]['bbox'][3]
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(anns[i]['category_id'])
                iscrowds.append(anns[i]['iscrowd'])
                masks.append(self.coco.annToMask(anns[i]))
                

            # # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.tensor(iscrowds, dtype=torch.uint8)
            
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
            target["masks"] = masks


        if self.data_transforms is not None:
            img, target = self.data_transforms(img, target)
        
        return img, target

    def __len__(self):
        return len(self.imgs)


def main():
    dataset = PennFudanDataset('dataset', train=False)
    for i in range(3):
        print(i)
        print(dataset.__getitem__(i))

if __name__ == '__main__':
    main()