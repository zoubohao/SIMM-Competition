import torch
import torch.nn as nn
import torch.nn.functional as F


def AddN(tensorList : []):
    if len(tensorList)==1:
        return tensorList[0]
    else:
        addR = tensorList[0] + tensorList[1]
        for i in range(2,len(tensorList)):
            addR = addR + tensorList[i]
        return addR


class Pool2dStaticSamePadding(nn.Module):
    """
    The real keras/tensorflow MaxPool2d with same padding
    """

    def __init__(self, kernel_size, stride ,pooling = "avg"):
        super().__init__()
        if pooling.lower() == "max":
            self.pool = nn.MaxPool2d(kernel_size=kernel_size,stride=stride)
        elif pooling.lower() == "avg":
            self.pool = nn.AvgPool2d(kernel_size=kernel_size,stride=stride,ceil_mode=True, count_include_pad=False)
        else:
            raise Exception("No implement.")
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)

        extra_h = h_cover_len - w
        extra_v = v_cover_len - h

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

import numpy as np
class Blur_Pooling(nn.Module):

    def __init__(self,in_channels,pooling_type = "Max"):
        super().__init__()
        self.stride = [2,2]
        self.kernel_size = [3,3]
        self.pooling = Pool2dStaticSamePadding(kernel_size=2,stride=1,pooling=pooling_type)
        bk = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]])
        bk = bk / np.sum(bk)
        bk = np.repeat(bk, in_channels)
        bk = np.reshape(bk, (in_channels,1,3,3))
        self.bk = nn.Parameter(torch.from_numpy(bk).float(),requires_grad=False)
        self.g = in_channels
        #print(self.bk)

    def forward(self,x):
        x = self.pooling(x)
        h, w = x.shape[-2:]
        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        extra_h = h_cover_len - w
        extra_v = v_cover_len - h
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = F.conv2d(x,self.bk,stride=[2,2],groups=self.g)
        return x




class Conv2dDynamicSamePadding(nn.Module):
    """
    The real keras/tensorflow conv2d with same padding
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        h_step = math.ceil(w / self.stride[1])
        v_step = math.ceil(h / self.stride[0])
        h_cover_len = self.stride[1] * (h_step - 1) + 1 + (self.kernel_size[1] - 1)
        v_cover_len = self.stride[0] * (v_step - 1) + 1 + (self.kernel_size[0] - 1)
        extra_h = h_cover_len - w
        extra_v = v_cover_len - h
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top
        x = F.pad(x, [left, right, top, bottom])
        x = self.conv(x)
        return x

class Blur_Down_Sample(nn.Module):

    def __init__(self,in_channels,out_channels,pooling_type):
        super().__init__()
        self.blur_pooling = Blur_Pooling(in_channels, pooling_type=pooling_type)
        self.convTrans = Conv2dDynamicSamePadding(in_channels,out_channels,1,bias=False)
        self.bn =  nn.BatchNorm2d(out_channels,eps=1e-3,momentum=1e-2)

    def forward(self,x):
        return self.bn(self.convTrans(self.blur_pooling(x)))


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

class SE(nn.Module):

    def __init__(self,in_channels,out_channels,reduce_factor = 2):
        super().__init__()
        self.dense1 = nn.Conv2d(in_channels,out_channels=in_channels // reduce_factor,kernel_size=1,
                                stride=1,padding=0)
        self.bn = nn.BatchNorm2d(in_channels // reduce_factor, eps=1e-3, momentum=1 - 0.99)
        self.dense2 = nn.Conv2d(in_channels // reduce_factor,out_channels=out_channels,kernel_size=1,
                                stride=1,padding=0)
        self.act = MemoryEfficientSwish()

    def forward(self,x):
        globalPooling = F.adaptive_avg_pool2d(x,[1,1])
        dense1 = self.act((self.bn(self.dense1(globalPooling))))
        dense2 = self.dense2(dense1)
        return torch.sigmoid(dense2)

from torch.nn import Parameter
class DynamicConv2d(nn.Module):

    def __init__(self,in_channels, out_channels,kernel_size, groups, stride, padding, tau = 1., k = 4):
        super().__init__()
        self.weight = Parameter(torch.ones(size=[k, out_channels, in_channels // groups, kernel_size, kernel_size],
                                           dtype=torch.float32,requires_grad=True),requires_grad=True)
        self.bias = Parameter(torch.zeros(size=[k, out_channels],requires_grad=True),requires_grad=True)
        self.stride = [stride,stride]
        self.padding = [padding,padding]
        self.dense1 = nn.Conv2d(in_channels, in_channels // 2,kernel_size=1,stride=1,padding=0)
        self.dense2 = nn.Conv2d(in_channels // 2, k ,kernel_size=1,stride=1,padding=0)
        self.tau = tau
        self.k = k
        self.groups = groups

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)

    def forward(self,x):
        globalAvg = F.adaptive_avg_pool2d(x,[1,1])
        dense1 = F.relu(self.dense1(globalAvg),True)
        dense2 = self.dense2(dense1)
        softmaxT = torch.softmax(torch.div(dense2,self.tau),dim=-3)
        batch = x.size(0)
        batchOut = []
        for i in range(batch):
            oneSample = x[i].unsqueeze(dim=0)
            softmaxTi = softmaxT[i]
            #print(softmaxTi)
            attentionW = softmaxTi.view(size=[self.k,1,1,1,1])
            attWeight = (self.weight * attentionW).sum(dim=0,keepdim=False)
            attentionB = softmaxTi.view(size=[self.k,1])
            attBias = (self.bias * attentionB).sum(dim=0,keepdim=False)
            batchOut.append(F.conv2d(oneSample, attWeight, attBias, self.stride,
                        self.padding, groups= self.groups))
        return torch.cat(batchOut,dim=0)


class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, norm= False, activation=False):
        super(SeparableConvBlock, self).__init__()

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dDynamicSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dDynamicSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation
        if self.activation:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.relu(x)
        return x




import matplotlib.pyplot as plt
### Check grad
def plot_grad_flow(named_parameters):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    ave_grads = []
    layers = []
    noneList = []
    for n, p in named_parameters:
        if p.requires_grad:
            layers.append(n)
            print("########")
            print(n)
            if p.grad is not None and "bias" not in n:
                print(p.grad.abs().mean())
                ave_grads.append(p.grad.abs().mean())
                # if p.grad.abs().mean() == 0:
                #     break
            else:
                print(None)
                noneList.append(n)
    print("Min grad : ",min(ave_grads))
    print("Max ",max(ave_grads))
    print(noneList)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

class FactorizedEmbedding(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size):
        """
        :param vocab_size:
        :param embed_size:
        :param hidden_size: hidden_size must much larger than embed_size
        """
        super(FactorizedEmbedding,self).__init__()
        self.embeddingLayer = nn.Embedding(vocab_size,embed_size)
        self.liner = nn.Sequential(nn.Linear(embed_size,hidden_size),
                                   nn.BatchNorm1d(hidden_size),
                                   Mish())

    def forward(self, x):
        """
        :param x: [batch,sequences]
        :return: [batch,sequences,hidden_size]
        """
        embedTensor = self.embeddingLayer(x)
        linerTensor = self.liner(embedTensor)
        return linerTensor

class Mish(nn.Module):

    def __init__(self):
        super(Mish,self).__init__()

    def forward(self,x):
        return x * torch.tanh(F.softplus(x))

class L2LossReg(nn.Module):

    def __init__(self,lambda_coefficient):
        super(L2LossReg,self).__init__()
        self.l = lambda_coefficient

    def forward(self,parameters):
        tensors = []
        for pari in parameters:
            name = pari[0].lower()
            tensor = pari[1]
            if "bias" not in name and "bn" not in name and "p" not in name:
                #print(name)
                tensors.append(torch.sum(torch.pow(tensor,2.)))
        return torch.mul(AddN(tensors),self.l)

def l2LossFunction(parameters,l):
    tensors = []
    for pari in parameters:
        name = pari[0].lower()
        tensor = pari[1]
        if "bias" not in name and "bn" not in name and "p" not in name:
            # print(name)
            tensors.append(torch.sum(torch.pow(tensor, 2.)))
    return torch.mul(AddN(tensors),l)

import math
import torch.utils.data
## Compute the mean and std value of dataset.
def get_mean_and_std(dataset):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    k = 0
    for inputs, sex, age_approx, anatom, target in dataloader:
        print(k)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
        k += 1
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def drop_connect_A(inputs, p, training):
    """ Drop connect. """
    if training is False: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def drop_connect_B(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(p=keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class MemoryEfficientSwish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return SwishImplementation.apply(x)


if __name__ == "__main__":
    testModel = Blur_Pooling(16)
    testInput = torch.ones(size=[5,16,32,32]).float()
    print(testModel(testInput).shape)