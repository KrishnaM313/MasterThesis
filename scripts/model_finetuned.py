import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models


class FintunedModel(nn.Module):
    def __init__(self, pretrainedModel, classes):
        super(FintunedModel, self).__init__()
        self.features = pretrainedModel.features
        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, classes),
            )


        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), 256 * 6 * 6)


        # if self.modelName == 'alexnet' :
        #     f = f.view(f.size(0), 256 * 6 * 6)
        # elif self.modelName == 'vgg16':
        #     f = f.view(f.size(0), -1)
        # elif self.modelName == 'resnet' :
        #     f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

