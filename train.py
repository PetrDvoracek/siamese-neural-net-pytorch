"""
Test the implementation by hand.
"""
import typing
from albumentations.augmentations.transforms import Normalize
import torch
import glob
import os
import cv2
import albumentations as A
import numpy as np

from icecream import ic
from albumentations.pytorch import ToTensorV2

from src.data import DataIterator, BatchIterator
from src.net import SiameseNet
from src.loss import triplet_loss
from src.train import train

DATASET_PATH = '/home/pedro/datasets/ffhq-faces/train/thumbnails128x128/'

if __name__ == '__main__':
    device = 'cuda'
    batch_size =32
    max_images = 30000
    resolution = 128
    max_images = 5000
    cv2_imgs = np.empty((max_images,resolution,resolution,3), dtype='uint8')

    counter = 0
    for folder in glob.glob(os.path.join(DATASET_PATH, '*/')):
        for file in glob.glob(folder+"*.png"):
            if counter>=max_images:
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
                # A.JpegCompression(p=0.5, quality_lower=70, quality_upper=100),
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


    net = SiameseNet().to(device)
    train(
        net=net,
        num_epochs=4,
        inputs=cv2_imgs,
        iterator=BatchIterator(
            batch_size=batch_size,
            preproc_f=preproc_f,
            augment_f=augment_f,
            max_images=max_images,
            device=device,
            ),
        loss_f=lambda x: triplet_loss(x, alpha=0.5),
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
    )