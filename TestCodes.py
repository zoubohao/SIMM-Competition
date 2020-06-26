import numpy as np
import torch
import torchvision as tv
from DataSet import SIMM_DataSet
from torch.utils.data import DataLoader
#
#
# def generator(data_loader):
#     while True:
#         for  (imgs, targets) in data_loader:
#             yield imgs,  targets
#
# def randomConcatTensor(t1s,t2s):
#     result = []
#     index = [i for i in range(t1s[0].shape[0] + t2s[0].shape[0])]
#     #np.random.shuffle(index)
#     if np.random.rand(1) > 0.5 :
#         for i, t in enumerate(t1s):
#             result.append(torch.cat([t, t2s[i]], dim=0)[index])
#     else:
#         for i, t in enumerate(t1s):
#             result.append(torch.cat([t2s[i], t], dim=0)[index])
#     return result
#
# # def randomConcatTensor(t1s,t2s):
# #     result = []
# #     if np.random.rand(1) > 0.5 :
# #         for i, t in enumerate(t1s):
# #             result.append(torch.cat([t, t2s[i]], dim=0))
# #     else:
# #         for i, t in enumerate(t1s):
# #             result.append(torch.cat([t2s[i], t], dim=0))
# #     return result
# #
# #
from PIL import Image

def generator(data_loader):
    while True:
        for (imgs, targets) in data_loader:
            yield imgs, targets


def randomConcatTensor(t1s, t2s):
    result = []
    index = [i for i in range(t1s[0].shape[0] + t2s[0].shape[0])]
    np.random.shuffle(index)
    lengT1s = len(t1s)
    lengT2s = len(t2s)
    assert lengT1s == lengT2s , "Two tensor lists must have same tensors. "
    for i in range(lengT1s):
        result.append( torch.cat([t1s[i], t2s[i]], dim=0)[index] )
    return result
batchSize = 6
transformationTrain = tv.transforms.Compose([
    tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)]),
    tv.transforms.CenterCrop(size=[272, 408]),
    tv.transforms.ToTensor(),
])

trainNegDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/trainNeg.csv", transforms=transformationTrain, train=True)
trainPosDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/trainPos.csv", transforms=transformationTrain, train=True)

trainNegDataLoader = DataLoader(trainNegDataSet, batch_size=batchSize // 2, shuffle=False, pin_memory=True)
trainPosDataLoader = DataLoader(trainPosDataSet, batch_size=batchSize // 2, shuffle=False, pin_memory=True)

negGene = generator(trainNegDataLoader)
posGene = generator(trainPosDataLoader)


for e in range(10):
    for times in range(100):
        imgsNegTr,  targetsNegTr = negGene.__next__()
        imgsPosTr,  targetsPosTr = posGene.__next__()
        imgsTr,  targetsTr = randomConcatTensor([imgsNegTr,  targetsNegTr],
                                                                          [imgsPosTr,  targetsPosTr])
        oneImage = tv.transforms.ToPILImage()(imgsTr[-1]).save("./Check.jpg")
        print("###########")
        print(targetsTr)
        break
    break



from PIL import Image





from Tools import DynamicConv2d
testInput = torch.ones(size=[10,64,32,32]).float()
testModel = DynamicConv2d(64,128,3,32,1,1,tau=10,k=2)
print(testModel(testInput).shape)

testa = np.zeros(shape=[3,5,5])
testa[:,2:4, 2:4] = 1
print(testa)

import torchvision as tv
def CutMix(image1, image2):
    imageW = 3096
    imageH = 2048
    ##
    img1 = tv.transforms.Resize([imageH, imageW])(image1)
    img2 = tv.transforms.Resize([imageH, imageW])(image2)
    la = np.random.randint(2,4)
    ##
    randx = np.random.randint(1,imageW // la * (la - 1) - 5)
    randy = np.random.randint(1,imageH // la * (la - 1) - 5)
    ##
    mask = np.zeros(shape=[1,imageH,imageW])
    mask[0,randy : (randy + imageH // la), randx : (randx + imageW // la)] = 1
    ##
    maskImg1 = tv.transforms.ToTensor()(img1) * mask
    maskImg2 = tv.transforms.ToTensor()(img2) * (1 - mask)
    ##
    addImg = (maskImg1 + maskImg2).float()
    return tv.transforms.ToPILImage()(addImg)

import  os
imgPIL1 = Image.open(os.path.join("./train","ISIC_0089738"+ ".jpg")).convert("RGB")
imgPIL2 = Image.open(os.path.join("./train","ISIC_0096227"+ ".jpg")).convert("RGB")
cutmixPIL = CutMix(imgPIL1,imgPIL2).save("./Check.jpg")

print([1,2,3] + [3,4,5,6])







