import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassHead(nn.Sequential):
    def __init__(self, in_features, num_classes):
        super(MultiClassHead, self).__init__(
            nn.Linear(in_features, num_classes)
        )
        
class MultiLabelHead(nn.Sequential):
    def __init__(self, in_features, num_classes):
        super(MultiLabelHead, self).__init__(
            nn.Linear(in_features, num_classes)
        )

class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(262144, num_classes)
        )
