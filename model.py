import torch
import torch.nn as nn
import torchvision.models as models

class WBNet(nn.Module):
    def __init__(self, meta_in):
        super().__init__()
        
        backbone = models.resnet18(pretrained=True)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_in, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, img, meta):
        img_feat = self.backbone(img)
        meta_feat = self.meta_mlp(meta)
        x = torch.cat([img_feat, meta_feat], dim=1)
        return self.head(x)
