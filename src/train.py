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
    loss_f: typing.Callable[[torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    ) -> None:
    for epoch in range(0, num_epochs):
        epoch_loss = 0
        for anchor, positive, negative in iterator(inputs):
            optimizer.zero_grad()

            prediction = net.forward(anchor, positive, negative)
            loss = loss_f(prediction)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            print(epoch_loss)
            # cv2.imshow('', anchor[62].swapaxes(0, -1).cpu().detach().numpy())
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()
            # cv2.imshow('', positive[62].swapaxes(0, -1).cpu().detach().numpy())
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()
            # cv2.imshow('', negative[62].swapaxes(0, -1).cpu().detach().numpy())
            # if cv2.waitKey(0) == ord('q'):
            #     cv2.destroyAllWindows()