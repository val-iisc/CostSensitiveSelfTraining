import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_BATCH_NORM_DECAY = 0.1
_BATCH_NORM_EPSILON = 1e-5


def batch_norm2d(num_features):
    return nn.BatchNorm2d(num_features, eps=_BATCH_NORM_EPSILON, momentum=_BATCH_NORM_DECAY)


def _weights_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight)


class IdentityBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = batch_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_norm2d(planes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = self.act(out)
        return out




class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False) 
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out

class build_ResNet:
    def __init__(self):
        print("Building resnet 50")
        pass
    def build(self, num_classes):
        return ResNet(num_classes)
