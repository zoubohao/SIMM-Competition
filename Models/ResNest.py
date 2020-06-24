import torch
import torch.nn as nn
import torch.nn.functional as F
from Tools import AddN
from Tools import Conv2dDynamicSamePadding
import math
from Tools import Pool2dStaticSamePadding
from Tools import Mish
from Tools import drop_connect_B
from Tools import SeparableConvBlock

class Split_Attention(nn.Module):

    def __init__(self,r,in_channels,inner_channels):
        super().__init__()
        self.r = r
        self.in_channels = in_channels
        self.dense1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,inner_channels,1,1),
                                    nn.BatchNorm2d(inner_channels,eps=1e-3,momentum=1e-2),
                                    Mish())
        self.dense2 = Conv2dDynamicSamePadding(inner_channels,in_channels * r,1,1,groups=r)


    def forward(self,inputs):
        addedTensor = AddN(inputs)
        globalPooling = F.adaptive_avg_pool2d(addedTensor,output_size=[1,1])
        dense1 = self.dense1(globalPooling)
        dense2List = torch.chunk(self.dense2(dense1),chunks=self.r,dim=-3)
        attentionList = []
        for i,oneDense in enumerate(dense2List):
            softMaxT = torch.softmax(oneDense,dim=1)
            attentionList.append(softMaxT * inputs[i])
        return AddN(attentionList)

class Cardinal_Block(nn.Module):

    def __init__(self,r,in_channels,drop_connect_rate):
        super().__init__()
        self.r = r
        ### Conv
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels,1,1,groups=r,bias=False),
                                   nn.GroupNorm(num_groups=r,num_channels=in_channels,eps=1e-3,affine=True),
                                   Mish())
        self.conv3 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels * r,3,1,groups=r,bias=False),
                                   nn.GroupNorm(num_groups=r,num_channels=in_channels * r,eps=1e-3,affine=True),
                                   Mish())
        ### Split
        self.split_attention = Split_Attention(r,in_channels,inner_channels=in_channels * 2)
        self.drop_conn = drop_connect_rate

    def forward(self,x):
        conv1 = self.conv1(x)
        conv3 = self.conv3(conv1)
        splitT = torch.chunk(conv3,self.r,dim=1)
        attenT = self.split_attention(splitT)
        if self.training and self.drop_conn > 0:
            attenT = drop_connect_B(attenT, self.drop_conn)
        return attenT

class OneBlock(nn.Module):

    def __init__(self,in_channels,out_channels,k = 2,r= 2,pooling = "AVG",drop_connect_rate = 0.5):
        super().__init__()
        self.k = k
        self.cardinalList = nn.ModuleList([Cardinal_Block(r, in_channels // k,drop_connect_rate) for _ in range(k)])
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,in_channels,1,1,bias=False),
                                   nn.BatchNorm2d(in_channels,eps=1e-3,momentum=1e-2))
        self.downSample = False
        if in_channels != out_channels:
            self.DownSample = nn.Sequential(
                Pool2dStaticSamePadding(2,2,pooling=pooling),
                Conv2dDynamicSamePadding(in_channels,out_channels,kernel_size=3,stride=1,bias=False),
                nn.BatchNorm2d(out_channels,eps=1e-3,momentum=1e-2))
            self.downSample = True


    def forward(self,x):
        oneCardinalT = torch.chunk(x,chunks=self.k,dim=-3)
        catList = []
        for i,cardinalM in enumerate(self.cardinalList):
            catList.append(cardinalM(oneCardinalT[i]))
        catTensor = self.conv1(torch.cat(catList,dim=-3))
        if self.downSample:
            addedTensor = catTensor + x.clone()
            return F.relu(self.DownSample(addedTensor),True)
        else:
            return F.relu(catTensor + x.clone(),True)

class Bottleneck(nn.Module):

    def __init__(self,in_channels,out_channels,layers,pooling = "avg",dropRate = 0.5):
        super().__init__()
        block = list()
        block.append(OneBlock(in_channels, out_channels, pooling=pooling,drop_connect_rate=dropRate))
        for i in range(layers - 1):
            block.append(OneBlock(out_channels, out_channels, pooling=pooling,drop_connect_rate=dropRate))
        self.blocks = nn.Sequential(*block)

    def forward(self,x):
        return self.blocks(x)


from Tools import FactorizedEmbedding
class ResNestNet(nn.Module):

    def __init__(self,in_channels,num_classes,dropRate,w,d):
        super().__init__()
        self.conv1 = nn.Sequential(Conv2dDynamicSamePadding(in_channels,64 * w,7,2,bias=False),
                                   nn.BatchNorm2d(64 * w,eps=1e-3,momentum=1e-2),
                                   Mish())
        self.b1 = Bottleneck(64 * w, 128 * w, 3 * d,pooling="MAX",dropRate=dropRate)
        self.b2 = Bottleneck(128 * w, 256 * w, 4 * d,pooling="AVG",dropRate=dropRate)
        self.b3 = Bottleneck(256 * w, 512 * w, 6 * d,pooling="AVG",dropRate=dropRate)
        self.b4 = Bottleneck(512 * w, 1024 * w, 3 * d,pooling="AVG",dropRate=dropRate)
        self.conv2 = SeparableConvBlock(1024 * w, 1024,norm=True, activation=True)
        self.fc1 = nn.Dropout(dropRate,True)
        self.fc2 = nn.Linear(1024, num_classes)
        self._initialize_weights()


    def forward(self,img):
        conv1 = self.conv1(img)
        #print(conv1.shape)
        b1 = self.b1(conv1)
        #print(b1.shape)
        b2 = self.b2(b1)
        #print(b2.shape)
        b3 = self.b3(b2)
        #print(b3.shape)
        b4 = self.b4(b3)
        #print(b4.shape)
        conv2 = self.conv2(b4)
        avgG = F.adaptive_avg_pool2d(conv2,[1,1]).view([-1,1024])
        return self.fc2(self.fc1(avgG))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



if __name__ == "__main__":
    import torch
    import torchvision as tv
    from DataSet import SIMM_DataSet
    from torch.utils.data import DataLoader
    from Tools import plot_grad_flow
    from PIL import Image
    transformationTrain = tv.transforms.Compose([
        tv.transforms.Resize([int(384 * 1.08) , int(576 * 1.08)],interpolation=Image.BICUBIC),
        tv.transforms.ToTensor(),
    ])
    testDataSet = SIMM_DataSet(root="./../train", csvFile="./../train.csv", transforms=transformationTrain, train=True)
    testDataLoader = DataLoader(testDataSet, batch_size=2, shuffle=True)
    testModule = ResNestNet(3, 2, 0.0, w=1, d=1)
    optimizer = torch.optim.SGD(testModule.parameters(), 5e-4, momentum=0.9, weight_decay=1e-5)
    lossCri = nn.CrossEntropyLoss(reduction="mean")

    for _, (imgs, targets) in enumerate(testDataLoader):
        ### img, sex, age_approx, anatom, target
        print(imgs.shape)
        print(targets)
        outputs = testModule(imgs)
        print(outputs)
        loss = lossCri(outputs, targets)
        loss.backward()
        plot_grad_flow(testModule.named_parameters())
        break










