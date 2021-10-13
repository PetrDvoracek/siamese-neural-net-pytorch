"""
Triplet loss implementation
"""
from typing import final
from albumentations.augmentations.crops.functional import clamping_crop
import torch

def triplet_loss(y_pred: torch.Tensor, alpha: float=0.5) -> torch.Tensor:
    total_length = y_pred.shape[-1]
    assert total_length % 3 == 0, f'total_length of value {total_length} is not divisible by three'

    anchor = y_pred[:, 0:int(total_length*1/3)]
    pos = y_pred[:, int(total_length*1/3):int(total_length*2/3)]
    neg = y_pred[:, int(total_length*2/3):]

    pos_distance = torch.sum(torch.square(anchor - pos), dim=1)
    neg_distance = torch.sum(torch.square(anchor - neg), dim=1)
    
    loss = pos_distance - neg_distance + alpha
    loss.clip_(0, 1e6)
    final_loss = torch.mean(loss)

    return final_loss

if __name__ == '__main__':
    from icecream import ic
    anchor = torch.randn(2, 4)
    positive = torch.randn(2, 4)
    negative = torch.randn(2, 4)
    y_predicted = torch.cat([anchor, positive, negative], dim=1)
    loss = triplet_loss(y_predicted)
    ic(type(loss))