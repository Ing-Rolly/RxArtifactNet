"""
model_torch.py
==============
RxArtifactNet avec backbone ResNet18 pré-entraîné (PyTorch).
Sorties : segmentation (12 canaux) + classification (12 classes).
Entrée : (N, 1, 512, 512) ou (N, 1, 256, 256).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from knowledge_base import CLASS_NAMES, NUM_CLASSES

assert NUM_CLASSES == 12

class ResUNetSegCls(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.enc1 = resnet.layer1  # 64
        self.enc2 = resnet.layer2  # 128
        self.enc3 = resnet.layer3  # 256
        self.enc4 = resnet.layer4  # 512

        # Blocs décodeur SANS upsample (on gère l'upsample avant concat)
        def dec(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        self.dec4 = dec(512 + 256, 256)
        self.dec3 = dec(256 + 128, 128)
        self.dec2 = dec(128 + 64,  64)
        self.dec1 = dec(64 + 64,   64)

        self.seg_head = nn.Conv2d(64, num_classes, 1)
        self.cls_fc = nn.Linear(512, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Encodeur
        x0 = self.enc0(x)          # 1/2
        x0 = self.pool(x0)         # 1/4
        s1 = self.enc1(x0)         # 1/4
        s2 = self.enc2(s1)         # 1/8
        s3 = self.enc3(s2)         # 1/16
        b  = self.enc4(s3)         # 1/32

        # Classification
        gap = b.mean(dim=(2, 3))          # (N, 512)
        cls_logits = self.cls_fc(gap)
        cls_probs = F.softmax(cls_logits, dim=1)

        # Segmentation – décodage avec upsamples manuels
        # d4 : upsampler b pour correspondre à s3, puis concat
        b_up = F.interpolate(b, size=s3.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([b_up, s3], dim=1))   # taille = s3 (1/16)

        # d3 : upsampler d4 pour correspondre à s2
        d4_up = F.interpolate(d4, size=s2.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d4_up, s2], dim=1))  # taille = s2 (1/8)

        # d2 : upsampler d3 pour correspondre à s1
        d3_up = F.interpolate(d3, size=s1.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d3_up, s1], dim=1))  # taille = s1 (1/4)

        # d1 : upsampler d2 pour correspondre à x0
        d2_up = F.interpolate(d2, size=x0.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d2_up, x0], dim=1))  # taille = x0 (1/4)

        # Tête de segmentation
        seg = torch.sigmoid(self.seg_head(d1))
        # Remonter à la résolution originale
        seg = F.interpolate(seg, size=x.shape[-2:], mode='bilinear', align_corners=False)

        return {
            "seg_map": seg,
            "cls_probs": cls_probs,
            "cls_logits": cls_logits,
            "features": gap
        }

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print(f"[ResUNetSegCls] Poids sauvegardés → {path}")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))
        print(f"[ResUNetSegCls] Poids chargés ← {path}")