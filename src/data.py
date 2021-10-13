import typing

import torch
import numpy as np

class DataIterator:
    def __call__(self, inputs: torch.Tensor) -> typing.Iterator:
        raise NotImplementedError

class BatchIterator(DataIterator):
    def __init__(
            self,
            batch_size: int,
            preproc_f: typing.Callable,
            augment_f: typing.Callable,
            max_images: int,
            device: str,
        ) -> None:
        super(BatchIterator, self).__init__()
        self.batch_size = batch_size
        self.augment_f = augment_f
        self.max_images = max_images
        self.preproc_f = preproc_f
        self.device = device
    
    def __call__(self, inputs: typing.Any) -> typing.Iterator:
        while True:
            res = inputs.shape[1:3] # image resolution
            anchors = torch.empty((self.batch_size, 3, res[0], res[1])).to(self.device)
            positives = torch.empty((self.batch_size, 3, res[0], res[1])).to(self.device)
            negatives = torch.empty((self.batch_size, 3, res[0], res[1])).to(self.device)
            for i in range(0, self.batch_size):
                while True:
                    index_anchor = torch.randint(low=0, high=self.max_images-1, size=(1,))
                    index_neg = torch.randint(low=0, high=self.max_images-1, size=(1,))
                    if index_anchor != index_neg:
                        break
                anchors[i] = self.preproc_f(inputs[index_anchor].copy())
                positives[i] = self.augment_f(inputs[index_anchor].copy())
                negatives[i] = self.preproc_f(inputs[index_neg].copy())
            yield [anchors, positives, negatives]

if __name__ == '__main__':
    from icecream import ic
    iterator = BatchIterator(
        batch_size=4,
        augment_f=lambda x: x*10,
        max_images=2,
    )
    data = torch.full((8, 3, 64, 64), fill_value=255)
    ic(data[0,:,0,0])
    for batch in iterator(data):
        ic(batch[0].shape)
        ic(batch[0][0,:,0,0])