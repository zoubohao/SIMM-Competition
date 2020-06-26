import torch
import torchvision as tv
from torch.utils.data import DataLoader
from Models.EfficientNet import EfficientNet
from DataSet import SIMM_DataSet
import torch.nn as nn
import numpy as np
import sklearn.metrics as metrics
from torch_optimizer import AdaBound



def generator(data_loader):
    while True:
        for  (imgs, targets) in data_loader:
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


if __name__ == "__main__":
    ### config
    batchSize = 10
    labelsNumber = 1
    epoch = 50
    displayTimes = 20
    reduction = 'mean'
    ###
    modelSavePath = "./Model_Weight/"
    saveTimes = 3500
    ###
    loadWeight = False
    trainModelLoad = "Model_EFB70.92.pth"
    ###
    LR = 1e-3
    ###
    device0 = "cuda:0"
    model_name = "b6"
    reg_lambda = 2e-4

    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        tv.transforms.CenterCrop(size=[272, 408]),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=90)], p=0.5),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transformationTest = tv.transforms.Compose([
        tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)]),
        tv.transforms.CenterCrop(size=[272, 408]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainNegDataSet = SIMM_DataSet(root="./NegTrainAugResize",csvFile="./CSVFile/augNeg.csv",
                                   transforms=transformationTrain,train=True)
    trainPosDataSet = SIMM_DataSet(root="./PosTrainAugResize", csvFile="./CSVFile/augPos.csv",
                                   transforms=transformationTrain, train=True)
    testDataSet = SIMM_DataSet(root="./train",csvFile="./CSVFile/val.csv",
                               transforms=transformationTest,train=True)

    trainNegDataLoader = DataLoader(trainNegDataSet,batch_size=batchSize // 2,shuffle=True,pin_memory=True)
    trainPosDataLoader = DataLoader(trainPosDataSet, batch_size= batchSize - batchSize // 2, shuffle=True, pin_memory=True)
    testloader = DataLoader(testDataSet, batch_size=1, shuffle=False)

    model = EfficientNet.from_pretrained("efficientnet-" + model_name,num_classes=labelsNumber,advprop=True).to(device0)
    print(model)

    negLength = trainNegDataSet.__len__()
    posLength = trainPosDataSet.__len__()
    print("Positive samples number ", posLength)
    print("Negative samples number ",negLength)
    trainTimesInOneEpoch = max(negLength,posLength) // (batchSize // 2) + 1

    lossCri = nn.BCELoss(reduction=reduction).to(device0)

    optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9, weight_decay=reg_lambda,nesterov=True)

    if loadWeight :
        model.load_state_dict(torch.load(modelSavePath + trainModelLoad))

    negGene = generator(trainNegDataLoader)
    posGene = generator(trainPosDataLoader)
    ### Train or Test

    model = model.train(mode=True)
    trainingTimes = 0
    print("Training %3d times in one epoch" % (trainTimesInOneEpoch,))
    for e in range(1,epoch + 1):

        for times in range(trainTimesInOneEpoch):
            imgsNegTr,  targetsNegTr = negGene.__next__()
            imgsPosTr,  targetsPosTr = posGene.__next__()
            imgsTr, targetsTr = randomConcatTensor(
                [imgsNegTr,  targetsNegTr],
                [imgsPosTr,  targetsPosTr])
            imagesCuda = imgsTr.to(device0, non_blocking=True)
            labelsCuda = targetsTr.float().to(device0, non_blocking=True)

            ## img, sex, age_approx, anatom
            optimizer.zero_grad()
            predict = torch.sigmoid(model(imagesCuda)).squeeze()
            criLoss = lossCri(predict, labelsCuda)
            criLoss.backward()
            optimizer.step()
            trainingTimes += 1

            if trainingTimes % displayTimes == 0:
                with torch.no_grad():
                    print("######################")
                    print("Epoch : %d , Training time : %d" % (e, trainingTimes))
                    print("Cri Loss is %.3f " % (criLoss.item()))
                    print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                    print("predicted labels : ", predict[0:5])
                    print("Truth labels : ", labelsCuda[0:5])


            if trainingTimes % saveTimes == 0:
                ### val part
                model = model.eval()
                test_loss = 0
                correct = 0
                total = 0
                with torch.no_grad():
                    targetsList = list()
                    scoreList = list()
                    for batch_idx, (imgsTe, targetsTe) in enumerate(testloader):
                        imgsTeCuda = imgsTe.to(device0,non_blocking=True)

                        ## img, sex, age_approx, anatom
                        probability = torch.sigmoid(model(imgsTeCuda)).squeeze()
                        probability = probability.detach().cpu().numpy()
                        scoreList.append(probability)
                        truth = targetsTe.numpy().squeeze()
                        targetsList.append(truth)
                        print(batch_idx)
                        print(probability)
                        print("Val part, " + str(probability) + " , " + str(truth))
                    fpr, tpr, _ = metrics.roc_curve(y_true=targetsList,y_score=scoreList,pos_label=1)
                    aucV = metrics.auc(fpr, tpr)
                torch.save(model.state_dict(), modelSavePath + "Model_EF" + model_name + str(aucV) + ".pth")
                model = model.train(mode=True)
    torch.save(model.state_dict(), modelSavePath + "Model_EF"+ model_name + "Final" + ".pth")





