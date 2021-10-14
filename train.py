"""
Training script.
"""
from albumentations.augmentations.transforms import Normalize
import torch
import glob
import os
import cv2
import albumentations as A
import numpy as np

from icecream import ic
from albumentations.pytorch import ToTensorV2

from src.data import BatchIterator
from src.net import SiameseNet
from src.loss import triplet_loss
from src.train import train

DATASET_PATH = '/home/pedro/datasets/ffhq-faces/train/thumbnails128x128/'
DEVICE = 'cuda'
BATCH_SIZE = 32
MAX_IMAGES = 30_000
RESOLUTION = 128

if __name__ == '__main__':
    cv2_imgs = np.empty((MAX_IMAGES,RESOLUTION,RESOLUTION,3), dtype='uint8')

    counter = 0
    for folder in glob.glob(os.path.join(DATASET_PATH, '*/')):
        for file in glob.glob(folder+"*.png"):
            if counter>=MAX_IMAGES:
                break
            cv2_imgs[counter] = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
            counter += 1

    def preproc_f(image):
        t = A.Compose([
            Normalize(),
            ToTensorV2(),
        ])
        augmented = t(image=image)
        return augmented['image']

    def augment_f(image):
        t = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(p=0.7, shift_limit=(-0.3, 0.3), scale_limit=(-0.3,0.3), rotate_limit=(-25, 25)),
                A.JpegCompression(p=0.5, quality_lower=70, quality_upper=100),
                A.HueSaturationValue(p=0.6),
                A.Blur(p=0.1),
                A.GaussNoise(p=0.3),
                A.RandomContrast(p=0.5),
                A.Normalize(),
                ToTensorV2(),
        ]
        )
        augmented = t(image=image)
        return augmented['image']


    net = SiameseNet().to(DEVICE)
    train(
        net=net,
        num_epochs=4,
        inputs=cv2_imgs,
        iterator=BatchIterator(
            batch_size=BATCH_SIZE,
            preproc_f=preproc_f,
            augment_f=augment_f,
            max_images=MAX_IMAGES,
            device=DEVICE,
            ),
        criterion=lambda x: triplet_loss(x, alpha=0.5),
        optimizer=torch.optim.SGD(net.parameters(), 1e-2, momentum=0.9),
    )