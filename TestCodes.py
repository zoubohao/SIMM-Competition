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
#
# def generator(data_loader):
#     while True:
#         for (imgs, targets) in data_loader:
#             yield imgs, targets
#
#
# def randomConcatTensor(t1s, t2s):
#     result = []
#     index = [i for i in range(t1s[0].shape[0] + t2s[0].shape[0])]
#     np.random.shuffle(index)
#     lengT1s = len(t1s)
#     lengT2s = len(t2s)
#     assert lengT1s == lengT2s , "Two tensor lists must have same tensors. "
#     for i in range(lengT1s):
#         result.append( torch.cat([t1s[i], t2s[i]], dim=0)[index] )
#     return result
# batchSize = 6
# transformationTrain = tv.transforms.Compose([
#     tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)]),
#     tv.transforms.CenterCrop(size=[272, 408]),
#     tv.transforms.ToTensor(),
# ])
#
# trainNegDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/trainNeg.csv", transforms=transformationTrain, train=True)
# trainPosDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/trainPos.csv", transforms=transformationTrain, train=True)
#
# trainNegDataLoader = DataLoader(trainNegDataSet, batch_size=batchSize // 2, shuffle=False, pin_memory=True)
# trainPosDataLoader = DataLoader(trainPosDataSet, batch_size=batchSize // 2, shuffle=False, pin_memory=True)
#
# negGene = generator(trainNegDataLoader)
# posGene = generator(trainPosDataLoader)
#
#
# for e in range(10):
#     for times in range(100):
#         imgsNegTr,  targetsNegTr = negGene.__next__()
#         imgsPosTr,  targetsPosTr = posGene.__next__()
#         imgsTr,  targetsTr = randomConcatTensor([imgsNegTr,  targetsNegTr],
#                                                                           [imgsPosTr,  targetsPosTr])
#         oneImage = tv.transforms.ToPILImage()(imgsTr[-1]).save("./Check.jpg")
#         print("###########")
#         print(targetsTr)
#         break
#     break
#
#
#
# from PIL import Image
#
#
#
#
#
# from Tools import DynamicConv2d
# testInput = torch.ones(size=[10,64,32,32]).float()
# testModel = DynamicConv2d(64,128,3,32,1,1,tau=10,k=2)
# print(testModel(testInput).shape)
#
# testa = np.zeros(shape=[3,5,5])
# testa[:,2:4, 2:4] = 1
# print(testa)
#
# import torchvision as tv
# def CutMix(image1, image2):
#     imageW = 3096
#     imageH = 2048
#     ##
#     img1 = tv.transforms.Resize([imageH, imageW])(image1)
#     img2 = tv.transforms.Resize([imageH, imageW])(image2)
#     la = np.random.randint(2,4)
#     ##
#     randx = np.random.randint(1,imageW // la * (la - 1) - 5)
#     randy = np.random.randint(1,imageH // la * (la - 1) - 5)
#     ##
#     mask = np.zeros(shape=[1,imageH,imageW])
#     mask[0,randy : (randy + imageH // la), randx : (randx + imageW // la)] = 1
#     ##
#     maskImg1 = tv.transforms.ToTensor()(img1) * mask
#     maskImg2 = tv.transforms.ToTensor()(img2) * (1 - mask)
#     ##
#     addImg = (maskImg1 + maskImg2).float()
#     return tv.transforms.ToPILImage()(addImg)
#
# import  os
# imgPIL1 = Image.open(os.path.join("./train","ISIC_0089738"+ ".jpg")).convert("RGB")
# imgPIL2 = Image.open(os.path.join("./train","ISIC_0096227"+ ".jpg")).convert("RGB")
# cutmixPIL = CutMix(imgPIL1,imgPIL2).save("./Check.jpg")
#
# print([1,2,3] + [3,4,5,6])


# previousJ = []
# for t in range(3):
#     j = np.random.randint(0, 3)
#     while j in previousJ:
#         j = np.random.randint(0, 3)
#     previousJ.append(j)
# print(previousJ)
#
#
# print(round(0.62131232112321312321112,4))
#
# print(0.9 == (1. - 0.1))
#
# print(torch.sigmoid(torch.as_tensor(8.)))

# from Models.MyEffiNet import ETNet
# # testInput = torch.randn(size=[5, 3, 224, 224]).float()
# # testModule = ETNet(1.5, 1.5 ,3, 0.5, 1, [224, 224])
# # print(testModule)
# # print(testModule(testInput).shape)


from Models.ETNet import ETNet
from Models.EfficientNet import EfficientNet
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
model = EfficientNet.from_pretrained("efficientnet-b5",num_classes=1)
# model = ETNet(w = 2, d = 1.5, expand_ratio = 3, drop_ratio = 0.5, classes_num=1,
#                   input_image_size=[224, 224])

model = model.train(True)
testInput = torch.randn(size=[8,3,512,224]).float()
writer.add_graph(model,testInput)
writer.close()
print(model(testInput).shape)


import torch.nn as nn
import torch.nn.functional as F
# testInput = torch.tensor(data=[-3,-2,-1,0,1,2,3,4,5,6]).float()
# print(torch.clamp(testInput, min= -1, max=5))
#
# testInput = torch.randn(size=[3,14,512]).float()
# testMo = nn.LayerNorm(512)
# result = testMo(testInput)
#
# print(result.shape)
# print((torch.abs(result) <= 0.1).float().sum())

# testInput1 = torch.randn(size=[1, 1, 5 , 5]).float()
# testInput2 = torch.randn(size=[1, 9, 1 , 1]).float()
# print((testInput1 * testInput2).shape)

# from Models.ETNet import PositionEncoding2D
# i = 28
# testM = PositionEncoding2D(i,i)
# print(testM.pe)
# pe = testM.pe.detach().cpu().numpy()
# print(len(np.unique(pe.reshape(i*i))))
# print(i * i)

# test1 = torch.tensor([[1,2],[3,4]]).float().view([1,1,2,2])
# test2 = torch.tensor([5,6]).float().view([1,2,1,1])
#
# print(test1)
# print(test2)
# print(test1 * test2)

