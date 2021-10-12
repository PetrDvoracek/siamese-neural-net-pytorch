"""
Siamese network implementation
"""

import torch
import torchvision

class SiameseNet(torch.nn.Module):
    def __init__(self) -> None:
        super(SiameseNet, self).__init__()
        pretrained = torchvision.models.resnet18(pretrained=True)
        base_features = torch.nn.ModuleList(pretrained.children())[:-1]
        self.backbone = torch.nn.Sequential(*base_features)

    def forward(self, anchor: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        out_anchor = self.backbone(anchor)
        out_pos = self.backbone(pos)
        out_neg = self.backbone(neg)
        return torch.cat([out_anchor, out_pos, out_neg], axis=1).squeeze().squeeze()


if __name__ == '__main__':
    net = SiameseNet()
    img = torch.randn(4, 3, 32, 32)
    print(net(img, img, img).shape)