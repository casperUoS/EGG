import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class DrawSender(nn.Module):
    def __init__(
            self,
            feat_size,
            hidden_size,
    ):
        super(DrawSender,self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size

        self.vgg16 = models.vgg16(pretrained=True)
