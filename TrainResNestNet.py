import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from Models.ResNeSt_Ori import  resnest50
from DataSet import SIMM_DataSet


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
    assert lengT1s == lengT2s, "Two tensor lists must have same tensors. "
    for i in range(lengT1s):
        result.append(torch.cat([t1s[i], t2s[i]], dim=0)[index])
    return result


if __name__ == "__main__":
    ### config
    batchSize = 10
    labelsNumber = 1
    epoch = 50
    displayTimes = 20
    reg_lambda = 2e-4
    reduction = 'mean'
    drop_rate = 0.25
    ###
    modelSavePath = "./Model_Weight/"
    saveTimes = 2500
    ###
    loadWeight = False
    trainModelLoad = 0.8641028597785978
    ###
    LR = 1.e-3
    ###
    device0 = "cuda:1"

    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=30)], p=0.5),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transformationTest = tv.transforms.Compose([
        tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainNegDataSet = SIMM_DataSet(root="./NegTrainAugResize", csvFile="./CSVFile/augNeg.csv",
                                   transforms=transformationTrain, train=True)
    trainPosDataSet = SIMM_DataSet(root="./PosTrainAugResize", csvFile="./CSVFile/augPos.csv",
                                   transforms=transformationTrain, train=True)
    testDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/val.csv",
                               transforms=transformationTest, train=True)

    trainNegDataLoader = DataLoader(trainNegDataSet, batch_size=batchSize // 2, shuffle=True, pin_memory=True)
    trainPosDataLoader = DataLoader(trainPosDataSet, batch_size=batchSize - batchSize // 2, shuffle=True,
                                    pin_memory=True)
    testloader = DataLoader(testDataSet, batch_size=1, shuffle=False)

    model = resnest50(number_classes=labelsNumber,drop_connect_ratio=drop_rate).to(device0)
    print(model)

    negLength = trainNegDataSet.__len__()
    posLength = trainPosDataSet.__len__()
    print("Positive samples number ", posLength)
    print("Negative samples number ", negLength)
    trainTimesInOneEpoch = max(negLength, posLength) // (batchSize // 2) + 1

    lossCri = nn.BCELoss(reduction=reduction).to(device0)

    optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9, weight_decay=reg_lambda,nesterov=True)

    if loadWeight:
        model.load_state_dict(torch.load(modelSavePath + "Model_Re" + str(trainModelLoad) + ".pth"))

    negGene = generator(trainNegDataLoader)
    posGene = generator(trainPosDataLoader)
    ### Train or Test

    model = model.train(mode=True)
    trainingTimes = 0
    print("Training %3d times in one epoch" % (trainTimesInOneEpoch,))
    for e in range(1, epoch + 1):

        for times in range(trainTimesInOneEpoch):
            imgsNegTr, targetsNegTr = negGene.__next__()
            imgsPosTr, targetsPosTr = posGene.__next__()
            imgsTr, targetsTr = randomConcatTensor(
                [imgsNegTr, targetsNegTr],
                [imgsPosTr, targetsPosTr])
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
                        imgsTeCuda = imgsTe.to(device0, non_blocking=True)

                        ## img, sex, age_approx, anatom
                        probability = torch.sigmoid(model(imgsTeCuda)).squeeze()
                        probability = probability.detach().cpu().numpy()
                        scoreList.append(probability)
                        truth = targetsTe.numpy().squeeze()
                        targetsList.append(truth)
                        print(batch_idx)
                        print(probability)
                        print("Val part, " + str(probability) + " , " + str(truth))
                    precision, recall, _ = metrics.precision_recall_curve(y_true=targetsList, probas_pred=scoreList,
                                                                          pos_label=1)
                    aucPR = round(metrics.auc(recall, precision), 4)
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=targetsList, y_score=scoreList, pos_label=1)
                    auc = round(metrics.auc(fpr, tpr), 4)
                torch.save(model.state_dict(), modelSavePath + "Model_Re_" + "AUC" + str(auc)
                               + "_AUCPR" + str(aucPR) + ".pth")
                model = model.train(mode=True)
    torch.save(model.state_dict(), modelSavePath + "Model_ReF" + ".pth")





