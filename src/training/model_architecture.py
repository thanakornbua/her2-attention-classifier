import torch
import torch.nn as nn
import torchvision.models as models


class MultiTaskLesionClassifier(nn.Module):
    """
    Multi-task CNN with shared backbone and two heads:
    - Classification head: 3 classes
    - Localization head: bbox [x, y, w, h] in normalized coords
    """

    def __init__(self, backbone: str = "resnet50", num_classes: int = 3):
        super().__init__()

        if backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            in_features = net.fc.in_features
            # remove final FC, keep avgpool
            self.backbone = nn.Sequential(*list(net.children())[:-1])  # outputs (B, 2048, 1, 1)
            feat_dim = 2048
        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = net.features
            feat_dim = net.classifier[1].in_features
        else:
            raise ValueError("Unsupported backbone: {}".format(backbone))

        # classification head
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

        # localization head
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Sigmoid(),  # ensure normalized in [0,1]
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W)
        feat = self.backbone(x)
        if feat.ndim == 4 and feat.shape[-2:] != (1, 1):
            # global avg pool if not already
            feat = nn.functional.adaptive_avg_pool2d(feat, (1, 1))
        cls_logits = self.cls_head(feat)
        loc_out = self.loc_head(feat)
        return cls_logits, loc_out
