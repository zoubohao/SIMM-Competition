
import torch
import torch.nn as nn
from Models.SplAtConv2d import SplAtConv2d
import torch.nn.functional as F

__all__ = ['ResNet', 'Bottleneck', "resnest101"]

class DropBlock2D(nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.prob = prob
        #print("Drop prob is ", self.prob)

    def forward(self,x):
        if self.training and self.prob > 0.:
            keep_ratio = 1.0 - self.prob
            mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
            mask.bernoulli_(p=keep_ratio)
            x.div_(keep_ratio)
            x.mul_(mask)
        return x


class Bottleneck(nn.Module):
    """ResNet Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 2
    def __init__(self, inplanes, planes, stride=1, downsample=None, cardinality=2, radix = 2,bottleneck_width=64, dilation=1,
                 norm_layer=None, dropblock_prob=0.25, last_gamma=False):
        super(Bottleneck, self).__init__()
        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        self.conv1 = nn.Conv2d(inplanes, group_width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(group_width)
        self.dropblock_prob = dropblock_prob
        self.expansion = 2

        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob)

        self.conv2 = SplAtConv2d(
                group_width, group_width, kernel_size=3,
                stride=stride, padding=dilation,
                dilation=dilation, groups=cardinality, bias=False,
                norm_layer=norm_layer,
                dropblock_prob=dropblock_prob,radix=radix)

        self.conv3 = nn.Conv2d(
            group_width, planes * self.expansion , kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)

        if last_gamma:
            from torch.nn.init import zeros_
            zeros_(self.bn3.weight)

        self.relu = nn.ReLU(True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        ### conv1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        ### conv2
        out = self.conv2(out)

        ### conv3
        out = self.conv3(out)
        out = self.bn3(out)

        if self.dropblock_prob > 0.0:
            out = self.dropblock(out)

        ### down sample
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

from Tools import Mish
from Tools import Conv2dDynamicSamePadding
class ResNet(nn.Module):

    def __init__(self, block, layers, groups=2, radix = 2, bottleneck_width=64,
                 num_classes=10,deep_stem=False, stem_width=64, avg_down=False, dropblock_prob=0.25,
                 last_gamma=False, norm_layer=nn.BatchNorm2d):

        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix

        super(ResNet, self).__init__()
        conv_layer = nn.Conv2d
        conv_kwargs =  {}
        if deep_stem:
            self.conv1 = nn.Sequential(
                Conv2dDynamicSamePadding(3, stem_width, kernel_size=5, stride=2),
                nn.BatchNorm2d(stem_width),
                Mish(),
                Conv2dDynamicSamePadding(stem_width, stem_width, kernel_size=3, stride=1,groups=stem_width),
                nn.GroupNorm(32,stem_width),
                Mish(),
                Conv2dDynamicSamePadding(stem_width,stem_width,kernel_size=1),
                nn.BatchNorm2d(stem_width)
            )
        else:
            self.conv1 = conv_layer(3, stem_width, kernel_size=7, stride=2, padding=3,bias=False, **conv_kwargs)
        self.reConv1 = Conv2dDynamicSamePadding(3,stem_width,3,2)

        self.conv2 = nn.Sequential(
            Conv2dDynamicSamePadding(stem_width,stem_width,kernel_size=3, stride=2, groups=stem_width, bias=False),
            nn.GroupNorm(num_groups=stem_width,num_channels=stem_width,eps=1e-3,affine=True),
            Mish(),
            Conv2dDynamicSamePadding(stem_width, stem_width,kernel_size=1, stride=1),
            nn.BatchNorm2d(stem_width)
        )
        self.reConv2 = Conv2dDynamicSamePadding(stem_width,stem_width,kernel_size=3,stride=2)

        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)

        self.dropout = nn.Dropout(0.4,True)
        self.fc1 = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,
                    dropblock_prob=0.25):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_layers = []
            if self.avg_down:
                down_layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride,
                                                    ceil_mode=True, count_include_pad=False))
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=1, bias=False))
            else:
                down_layers.append(nn.Conv2d(self.inplanes, planes * block.expansion,
                                             kernel_size=1, stride=stride, bias=False))

            down_layers.append(norm_layer(planes * block.expansion))
            downsample = nn.Sequential(*down_layers)

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, cardinality=self.cardinality, radix = self.radix,
                            bottleneck_width=self.bottleneck_width,
                            dilation=1,
                            norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                            last_gamma=self.last_gamma))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality=self.cardinality, radix = self.radix,
                                bottleneck_width=self.bottleneck_width,
                                dilation=1,
                                norm_layer=norm_layer, dropblock_prob=dropblock_prob,
                                last_gamma=self.last_gamma))

        return nn.Sequential(*layers)


    def forward(self, img):
        conv1 = self.conv1(img) +  self.reConv1(img)
        #print(x.shape)
        conv2 = self.conv2(conv1) + self.reConv2(conv1)
        #print(x.shape)
        x = self.layer1(conv2)
        #print(x.shape)
        x = self.layer2(x)
        #print(x.shape)
        x = self.layer3(x)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)

        fc1 = self.fc1(self.dropout(F.adaptive_avg_pool2d(x,[1,1]).view([-1,1024])))

        return fc1

def resnest101(number_classes, drop_connect_ratio):
    model = ResNet(Bottleneck, [3, 4, 17, 3], groups=2, radix = 2, bottleneck_width=64,
                   deep_stem=True, stem_width=64, avg_down=True,
                   dropblock_prob=drop_connect_ratio,
                   num_classes=number_classes,last_gamma=True)
    return model

if __name__ == "__main__":
    imgs = torch.rand(size=[10,3,272, 408])
    testModule = resnest101(2, 0.4)
    outputs = testModule(imgs)
    print(outputs)