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
