"""
Siamese network training pipeline.
"""

import typing
import torch
import cv2

from src.data import DataIterator, BatchIterator
from src.net import SiameseNet
from src.loss import triplet_loss

def train(
    net: SiameseNet,
    inputs: torch.Tensor,
    num_epochs: int,
    iterator: DataIterator,
    criterion: typing.Callable[[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    ) -> None:
    for epoch in range(0, num_epochs):
        epoch_loss = 0.0
        for idx, (anchor, positive, negative) in enumerate(iterator(inputs)):
            optimizer.zero_grad()

            prediction = net.forward(anchor, positive, negative)
            loss = criterion(prediction)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if idx % 10 == 9:
                print(f'step {idx} loss {epoch_loss / idx:0.3f}')