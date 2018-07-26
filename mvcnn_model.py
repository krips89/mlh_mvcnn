import matplotlib.pyplot as plt
from config import *
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import torchvision.models as models


import warnings
warnings.filterwarnings("ignore")

plt.ion()

model = models.vgg16_bn(pretrained=True)
#
def load_vgg():
    vgg_model = models.vgg16_bn(pretrained=True)
    n_conv = nn.Conv2d(5, 64, kernel_size=3, padding=1)
    trained_kernel = (vgg_model._modules['features'][0].weight)
    feature = nn.Sequential(*list(vgg_model.features.children())[-42:])
    for i in range(3):
        n_conv.weight.data[:, i, :, :] = trained_kernel.data[:, i, :, :]
    n_conv.weight.data[:,3,:,:] = (trained_kernel.data[:, 0, :, :] + trained_kernel.data[:, 1, :, :])/2
    n_conv.weight.data[:,4,:,:] = (trained_kernel.data[:, 1, :, :] + trained_kernel.data[:, 2, :, :])/2
    return nn.Sequential(n_conv, feature)

class MVConv(nn.Module):
    def __init__(self):
        super(MVConv, self).__init__()
        self.features1 = load_vgg()
        self.features2 = load_vgg()
        self.features3 = load_vgg()
        self.features4  = nn.Sequential(nn.Conv2d(1536, 512, kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(True),
                                        nn.MaxPool2d(kernel_size=2,stride=1))

        self.classifier = nn.Sequential(model.classifier,
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x, y, z):

        x1 = self.features1(x)
        x2 = self.features2(y)
        x3 = self.features3(z)

        combined = torch.cat((x1,x2,x3),1)
        xcat = self.features4(combined)
        x = xcat.view(xcat.size(0), -1)

        x = self.classifier(x)

        return x
