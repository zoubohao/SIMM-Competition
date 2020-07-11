import torch
from torch import nn
from torch.nn import functional as F
from .utils import Conv2dDynamicSamePadding
from .utils import drop_connect
from .utils import MemoryEfficientSwish
import math

import math
# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
# This module is copied from Github.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.2).
        max_len: the max. length of the incoming sequence.
    """

    def __init__(self, d_model, max_len, dropout=0.2):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

import numpy as np
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
        pe = torch.from_numpy(pe).float().unsqueeze(0).unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self,x):
        """
        :param x: [b, c, h, w]
        :return: [b, c, h, w]
        """
        #print(self.pe.shape)
        return x + self.pe


class LinearTanh(nn.Module):

    def __init__(self, minV, maxV):
        super().__init__()
        k = 2. / (maxV - minV)
        b = 1 - maxV * k
        k = torch.tensor(k).float()
        b = torch.tensor(b).float()
        self.register_buffer("k", k)
        self.register_buffer("b", b)
        self.maxV = maxV
        self.minV = minV

    def forward(self, x):
        x = torch.clamp(x, self.minV, self.maxV)
        return x * self.k + self.b

class Attention(nn.Module):

    def __init__(self,inChannels, reducedHeight, reducedWidth, drop_p = 0.1):
        """
        :param inChannels: input channels
        :param reducedHeight: the reduced height
        :param reducedWidth: the reduced width
        :param drop_p: drop probability
        """

        super().__init__()
        ### pixel attention
        self.reducedHeight = reducedHeight
        self.reducedWidth = reducedWidth
        self.position1D = PositionalEncoding(inChannels, self.reducedHeight * self.reducedWidth, drop_p)
        self.position2D = PositionEncoding2D(reducedHeight, reducedWidth)
        self.pixelAttention = nn.MultiheadAttention(inChannels, 1, drop_p)
        self.linear = nn.Linear(inChannels,inChannels)
        self.ln = nn.LayerNorm(inChannels)
        ### channels attention
        num_squeezed_channels = max(1, inChannels // 4)
        self._se_reduce = nn.Conv2d(in_channels=inChannels, out_channels=num_squeezed_channels, kernel_size=1)
        self._se_bn = nn.BatchNorm2d(num_features= num_squeezed_channels, momentum= 1 - 0.99, eps=1e-3)
        self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=inChannels, kernel_size=1)
        self._swishc = MemoryEfficientSwish()
        ### fusion
        self.conv = nn.Sequential(nn.BatchNorm2d(num_features= inChannels, momentum= 1 - 0.99, eps=1e-3),
                                  nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1))

        ### [-1, 1] hard linear tanh
        self.linearTanh = LinearTanh(-6, 6)

    def forward(self,x):
        b, c, h, w = x.shape
        #print("Input x shape is {}".format(x.shape))
        #print("The reduced shape is {}".format([self.reducedWidth, self.reducedHeight]))
        ### pixel
        if self.reducedHeight != h or self.reducedWidth != w:
            avgT = F.adaptive_avg_pool2d(x, [self.reducedHeight, self.reducedWidth])
            oneDTensor = torch.reshape(avgT, [b, self.reducedHeight * self.reducedWidth, c]).transpose(0,1)
            oneDTensor = self.position1D(oneDTensor)
            pixelAttention, _ = self.pixelAttention(oneDTensor, oneDTensor, oneDTensor, need_weights=False)
            oriShape = pixelAttention.transpose(0, 1)
            oriShape = self.ln(oriShape)
            oriShape = self.linear(oriShape)
            oriShape = torch.reshape(oriShape, [-1, c, self.reducedHeight, self.reducedWidth])
            oriShape = self.position2D(oriShape)
            oriShape = F.interpolate(oriShape, size=[h, w], mode="bilinear", align_corners=True)
            #print("Not same shape")
        else:
            oneDTensor = torch.reshape(x, [b, h * w, c]).transpose(0,1)
            oneDTensor = self.position1D(oneDTensor)
            pixelAttention, _ = self.pixelAttention(oneDTensor, oneDTensor, oneDTensor, need_weights=False)
            oriShape = pixelAttention.transpose(0, 1)
            oriShape = self.ln(oriShape)
            oriShape = self.linear(oriShape)
            oriShape = torch.reshape(oriShape, [-1, c, h, w])
            oriShape = self.position2D(oriShape)
            #print("Same shape")
        ### channels
        x_squeezed = F.adaptive_avg_pool2d(x, [1,1])
        x_squeezed = self._se_reduce(x_squeezed)
        x_squeezed = self._se_bn(x_squeezed)
        x_squeezed = self._swishc(x_squeezed)
        x_squeezed = self._se_expand(x_squeezed)
        ### multiply [b, c, h, w]
        attent = oriShape * x_squeezed
        attent = self.conv(attent)
        return self.linearTanh(attent)


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
        self.atten = Attention(oup, reducedHeight, reducedWidth)

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
        # values in this mask are between [0, 1]
        # [0, 1] * X + X ---> [1 , 2] * x
        # mask * x + x
        x = self.atten(x) * x + x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

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
        self.seq = nn.Sequential(Conv2dDynamicSamePadding(inChannels,outChannels, kSize, stride, bias=False),
                                 nn.BatchNorm2d(num_features=outChannels, momentum=1 - 0.99, eps=1e-3),
                                 MemoryEfficientSwish())
    def forward(self, x):
        return self.seq(x)


class BottleBlock (nn.Module):

    def __init__(self, inChannels, outChannels, kernel,  expand_ratio, drop_ratio, layers, stride, reducedHeight, reducedWidth):
        super().__init__()
        self.restMBs = nn.ModuleList()
        self.restMBs.add_module("TransCMB",MBConvBlock(inChannels, outChannels, kernel, expand_ratio,
                                                       drop_ratio = drop_ratio, reducedHeight=reducedHeight,
                                                       reducedWidth=reducedWidth,skip = False, stride = stride))
        for t in range(layers - 1):
            self.restMBs.add_module(name="MBConv" + str(t), module=MBConvBlock(outChannels, outChannels, kernel, expand_ratio, drop_ratio,
                                                                               reducedHeight=reducedHeight,
                                                                               reducedWidth=reducedWidth))

    def forward(self, x):
        for block in self.restMBs:
            x = block(x)
        return x

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
        self.convIni = nn.Sequential(Conv2dDynamicSamePadding(3, int(w * 32), 7, 2, bias=False),
                                     nn.BatchNorm2d(num_features=int(w * 32), momentum=1 - 0.99, eps=1e-3),
                                     MemoryEfficientSwish())
        self.layer1 = BottleBlock(int(w * 32), int(w * 16), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 1),  stride = 1,reducedHeight=height // 4, reducedWidth=width // 4)
        self.trans1 = TransBlock(3, int(w * 16), 7, 2)
        ### 2 56
        self.layer2 = BottleBlock(int(w * 16), int(w * 24), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=2,reducedHeight=height // 8, reducedWidth=width // 8)
        self.trans2 = TransBlock(int(w * 16), int(w * 24) , 5, 2)
        ### 2 28
        self.layer3 = BottleBlock(int(w * 24), int(w * 40), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 2),  stride=2,reducedHeight=height // 16, reducedWidth=width // 16)
        self.trans3 = TransBlock(int(w * 24), int(w * 40), 5, 2)
        ### 2 14
        self.layer4 = BottleBlock(int(w * 40), int(w * 80), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 3),  stride=2,reducedHeight=height // 32, reducedWidth=width // 32)
        self.layer5 = BottleBlock(int(w * 80), int(w * 112), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 3),  stride=1,reducedHeight=height // 32, reducedWidth=width // 32)
        self.trans4 = TransBlock(int(w * 40), int(w * 112), 3, 2)
        ### 2 7
        self.layer6 = BottleBlock(int(w * 112), int(w * 192), 5,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 4),  stride=2, reducedHeight=height // 32, reducedWidth=width // 32)
        self.layer7 = BottleBlock(int(w * 192), int(w * 320), 3,  expand_ratio, drop_ratio=drop_ratio,
                                  layers=int(d * 1),  stride=1, reducedHeight=height // 32, reducedWidth=width // 32)
        ###
        self.convFinal = nn.Sequential(Conv2dDynamicSamePadding(int(w * 320), 1280, 3, bias=False),
                                       nn.BatchNorm2d(1280, momentum=1 - 0.99, eps=1e-3),
                                       MemoryEfficientSwish())
        self.dropout = nn.Dropout(drop_ratio + 0.3, True)
        self.classify = nn.Linear(1280, classes_num)


    def forward(self, inputs):
        #print(inputs.shape)
        convIni = self.convIni(inputs)
        #print(convIni.shape)
        layer1 = self.layer1(convIni) + self.trans1(inputs)
        #print(layer1.shape)
        layer2 = self.layer2(layer1) + self.trans2(layer1)
        #print(layer2.shape)
        layer3 = self.layer3(layer2) + self.trans3(layer2)
        #print(layer3.shape)
        layer4 = self.layer4(layer3)
        #print(layer4.shape)
        layer5 = self.layer5(layer4) + self.trans4(layer3)
        #print(layer5.shape)
        layer6 = self.layer6(layer5)
        #print(layer6.shape)
        layer7 = self.layer7(layer6)
        #print(layer7.shape)
        ### final
        convF = self.convFinal(layer7)
        globalTen = F.adaptive_avg_pool2d(convF,[1,1]).view([-1, 1280])
        return self.classify(self.dropout(globalTen))


