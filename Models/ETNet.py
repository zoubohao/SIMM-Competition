import torch
from torch import nn
from torch.nn import functional as F
from .utils import Conv2dDynamicSamePadding
from .utils import drop_connect
from .utils import MemoryEfficientSwish
import math
import numpy as np
from Tools import SE


class PositionEncoding2D(nn.Module):

    def __init__(self,reducedHeight, reducedWidth):
        super().__init__()
        pe = np.zeros(shape=[reducedHeight, reducedWidth])
        r = 0.
        c = 1.
        for i in range(reducedHeight):
            for j in range(reducedWidth):
                if (i * j) % 2 == 0:
                    pe[i,j] = math.cos((i ** 2.7 + math.pow(j, 0.38)) * math.sin( r / (i + 1.)))
                    r += 2
                else:
                    #print("c is {}, j is {}".format(c,j))
                    pe[i,j] = math.sin(i ** 2.15 + math.pow(j, 0.4) * math.cos(- c / (j + 1.)))
                    c += 2
        pe = torch.from_numpy(pe).float().unsqueeze(0).unsqueeze(0).reshape([1, reducedHeight * reducedWidth , 1])
        self.register_buffer("pe", pe)

    def forward(self,x):
        """
        :param x: [b,  h * w , c]
        :return: [b, h * w, c]
        """
        #print(self.pe.shape)
        return x + self.pe


class AttentionAugmentedConvolution(nn.Module):

    def __init__(self, inChannels, reducedHeight, reducedWidth, drop_p = 0.1):
        super().__init__()
        ### transformer attention block
        self.reducedHeight = reducedHeight
        self.reducedWidth = reducedWidth
        self.transformer = nn.MultiheadAttention(inChannels, 2, drop_p)
        self.position2D = PositionEncoding2D(reducedHeight, reducedWidth)
        ### conv block
        self.conv = Conv2dDynamicSamePadding(inChannels, inChannels, 3, 1, bias=False)
        self.bn = nn.BatchNorm2d(inChannels, eps=1e-3, momentum=1 - 0.99)
        ###
        self.convF = nn.Sequential(nn.Conv2d(2 * inChannels, inChannels, kernel_size=1, stride=1),
                                   nn.BatchNorm2d(inChannels, eps=1e-3, momentum=1 - 0.99),
                                   MemoryEfficientSwish())

    def forward(self,x):
        b, c, h, w = x.shape
        #print("Input x shape is {}".format(x.shape))
        #print("The reduced shape is {}".format([self.reducedWidth, self.reducedHeight]))
        ### transformer branch
        if self.reducedHeight != h or self.reducedWidth != w:
            avgT = F.adaptive_avg_pool2d(x, [self.reducedHeight, self.reducedWidth])
            oneDTensor = torch.reshape(avgT, [b, self.reducedHeight * self.reducedWidth, c])
            oneDTensor = self.position2D(oneDTensor).transpose(0,1)
            pixelAttention, _ = self.transformer(oneDTensor, oneDTensor, oneDTensor, need_weights=False)
            oriShape = torch.reshape(pixelAttention.transpose(0,1), [-1, c , self.reducedHeight, self.reducedWidth])
            oriShape = F.interpolate(oriShape, size=[h, w], mode="bilinear", align_corners=True)
            # print("Not same shape")
        else:
            oneDTensor = torch.reshape(x, [b, self.reducedHeight * self.reducedWidth, c])
            oneDTensor = self.position2D(oneDTensor).transpose(0,1)
            pixelAttention, _ = self.transformer(oneDTensor, oneDTensor, oneDTensor, need_weights=False)
            oriShape = torch.reshape(pixelAttention.transpose(0,1), [-1, c , self.reducedHeight, self.reducedWidth])
            # print("Same shape")
        ### conv branch
        convB = self.bn(self.conv(x))
        ### concat
        concat = torch.cat([oriShape, convB], dim=-3)
        return self.convF(concat)



class MBConvBlock(nn.Module):

    """
    Mobile Inverted Residual Bottleneck Block.
    """

    def __init__(self, inChannels, outChannels, kernel, expand_ratio, drop_ratio, reducedHeight, reducedWidth, skip = True, stride = 1):
        super().__init__()
        self._bn_mom = 1 - 0.99 # pytorch's difference from tensorflow
        self._bn_eps = 1e-3
        self.drop_ratio = drop_ratio

        # Expansion phase (Inverted Bottleneck)
        inp = inChannels  # number of input channels
        oup = inChannels * expand_ratio  # number of output channels
        self._expand_conv = Conv2dDynamicSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = kernel
        self.id_skip = skip
        s = stride
        self._depthwise_conv = Conv2dDynamicSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Attention layer
        self.attenConv = AttentionAugmentedConvolution(outChannels, reducedHeight, reducedWidth)
        self.se = SE(oup, oup, reduce_factor=4)

        # Pointwise convolution phase
        final_oup = outChannels
        self._project_conv = Conv2dDynamicSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish1 = MemoryEfficientSwish()
        self._swish2 = MemoryEfficientSwish()

    def forward(self, inputs):
        """MBConvBlock's forward function.
        Args:
            inputs (tensor): Input tensor.
        Returns:
            Output of this block after processing.
        """
        # Expansion and Depthwise Convolution
        x = self._expand_conv(inputs)
        x = self._bn0(x)
        x = self._swish1(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish2(x)

        # attention
        x = self.se(x) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        x = self.attenConv(x)

        # drop connect
        if self.drop_ratio > 0:
            x = drop_connect(x, p=self.drop_ratio, training=self.training)

        # skip connection
        if self.id_skip :
            xCopy = inputs.clone()
            x = x + xCopy

        return x

class TransBlock(nn.Module):

    def __init__(self, inChannels, outChannels, kSize,  stride):
        super().__init__()
        self.conv = Conv2dDynamicSamePadding(inChannels,outChannels, kSize, stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=outChannels, momentum=1 - 0.99, eps=1e-3)

    def forward(self, x):
        return self.bn(self.conv(x))


class BottleBlock (nn.Module):

    def __init__(self, inChannels, outChannels, kernel,  expand_ratio, drop_ratio, layers, stride, reducedHeight, reducedWidth):
        super().__init__()
        self.restMBs = nn.ModuleList()
        self.trans = TransBlock(inChannels, outChannels, 3, stride)
        self.restMBs.add_module("TransCMB",MBConvBlock(inChannels, outChannels, kernel, expand_ratio,
                                                       drop_ratio = drop_ratio, reducedHeight=reducedHeight,
                                                       reducedWidth=reducedWidth,skip = False, stride = stride))
        for t in range(layers - 1):
            self.restMBs.add_module(name="MBConv" + str(t), module=MBConvBlock(outChannels, outChannels, kernel, expand_ratio, drop_ratio,
                                                                               reducedHeight=reducedHeight,
                                                                               reducedWidth=reducedWidth))

    def forward(self, x):
        xCopy = x.clone()
        for block in self.restMBs:
            x = block(x)
        return x + self.trans(xCopy)

class ETNet (nn.Module):

    def __init__(self, w, d, expand_ratio, drop_ratio,classes_num,input_image_size):
        """
        :param w: coefficient of channels
        :param d: coefficient of layers
        :param expand_ratio: expand ratio of channels of each layers
        :param drop_ratio: drop rate
        :param input_image_size: The image size must be divided by 32. [height, width]
        """
        super().__init__()
        ### 2 112
        height, width = input_image_size
        self.convIni = nn.Sequential(Conv2dDynamicSamePadding(3, int(w * 32), 3, 2, bias=False),
                                     nn.BatchNorm2d(num_features=int(w * 32), momentum=1 - 0.99, eps=1e-3),
                                     MemoryEfficientSwish())
        self.layer1 = BottleBlock(int(w * 32), int(w * 16), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 1),  stride = 1,reducedHeight=height // 4, reducedWidth=width // 4)
        ### 2 56
        self.layer2 = BottleBlock(int(w * 16), int(w * 24), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=2,reducedHeight=height // 8, reducedWidth=width // 8)
        ### 2 28
        self.layer3 = BottleBlock(int(w * 24), int(w * 40), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 3),  stride=2,reducedHeight=height // 16, reducedWidth=width // 16)
        ### 2 14
        self.layer4 = BottleBlock(int(w * 40), int(w * 80), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=2,reducedHeight=height // 32, reducedWidth=width // 32)
        self.layer5 = BottleBlock(int(w * 80), int(w * 112), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 3),  stride=1,reducedHeight=height // 32, reducedWidth=width // 32)
        ### 2 7
        self.layer6 = BottleBlock(int(w * 112), int(w * 192), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=2, reducedHeight=height // 32, reducedWidth=width // 32)
        self.layer7 = BottleBlock(int(w * 192), int(w * 320), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=1, reducedHeight=height // 32, reducedWidth=width // 32)
        ###
        self.convFinal = nn.Sequential(Conv2dDynamicSamePadding(int(w * 320), 2048, 3, bias=False),
                                       nn.BatchNorm2d(2048, momentum=1 - 0.99, eps=1e-3),
                                       MemoryEfficientSwish())
        self.dropout = nn.Dropout(drop_ratio + 0.35, True)
        self.classify = nn.Linear(2048, classes_num)


    def forward(self, inputs):
        #print(inputs.shape)
        convIni = self.convIni(inputs)
        #print(convIni.shape)
        layer1 = self.layer1(convIni)
        #print(layer1.shape)
        layer2 = self.layer2(layer1)
        #print(layer2.shape)
        layer3 = self.layer3(layer2)
        #print(layer3.shape)
        layer4 = self.layer4(layer3)
        #print(layer4.shape)
        layer5 = self.layer5(layer4)
        #print(layer5.shape)
        layer6 = self.layer6(layer5)
        #print(layer6.shape)
        layer7 = self.layer7(layer6)
        #print(layer7.shape)
        ### final
        convF = self.convFinal(layer7)
        globalTen = F.adaptive_avg_pool2d(convF,[1,1]).view([-1, 2048])
        return self.classify(self.dropout(globalTen))


