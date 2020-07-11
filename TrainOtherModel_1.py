import numpy as np
import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torchvision as tv
from torch.utils.data import DataLoader
from Models.SENet import se_resnext101_32x4d
from DataSet import SIMM_DataSet
from WarmUpSch import GradualWarmupScheduler


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
    ## The value of alpha must less than 0.5
    alpha = 0.0
    ### config
    batchSize = 12
    labelsNumber = 1
    epoch = 10
    displayTimes = 20
    reg_lambda = 2.5e-4
    reduction = 'mean'
    drop_rate = 0.6
    ###
    modelSavePath = "./Model_Weight/"
    saveTimes = 2500
    ###
    loadWeight = True
    trainModelLoad = "Model_Oth_AUC0.8943_AUCPR0.1564.pth"
    if_Ori = False
    ###
    LR = 1e-4
    warmEpoch = 2
    multiplier = 5
    ###
    device0 = "cuda:0"

    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        tv.transforms.RandomCrop([224, 224]),
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=30)], p=0.25),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transformationTest = tv.transforms.Compose([
        tv.transforms.Resize([256, 256]),
        tv.transforms.CenterCrop([224, 224]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    trainNegDataSet = SIMM_DataSet(root="./NegTrainAugResize", csvFile="./CSVFile/augNeg.csv",
                                   transforms=transformationTrain, train=True, alpha = alpha)
    trainPosDataSet = SIMM_DataSet(root="./PosTrainAugResize", csvFile="./CSVFile/augPos.csv",
                                   transforms=transformationTrain, train=True, alpha = alpha)
    testDataSet = SIMM_DataSet(root="./train", csvFile="./CSVFile/val.csv",
                               transforms=transformationTest, train=True, alpha = alpha)

    trainNegDataLoader = DataLoader(trainNegDataSet,batch_size=batchSize // 2,shuffle=True,pin_memory=True, num_workers=2)
    trainPosDataLoader = DataLoader(trainPosDataSet, batch_size= batchSize - batchSize // 2, shuffle=True, pin_memory=True, num_workers=2)
    testloader = DataLoader(testDataSet, batch_size=1, shuffle=False, num_workers=2)

    ### DenseNet-169
    model = se_resnext101_32x4d(num_classes=labelsNumber,dropout=drop_rate).to(device0)
    print(model)

    negLength = trainNegDataSet.__len__()
    posLength = trainPosDataSet.__len__()
    print("Positive samples number ", posLength)
    print("Negative samples number ", negLength)
    trainTimesInOneEpoch = max(negLength, posLength) // (batchSize // 2) + 1

    lossCri = nn.BCELoss(reduction=reduction).to(device0)

    optimizer = torch.optim.SGD(model.parameters(),lr=LR,momentum=0.9, weight_decay=reg_lambda,nesterov=True)
    #optimizer = torch.optim.Adam(model.parameters(), lr=LR,  weight_decay=reg_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch, after_scheduler=cosine_scheduler)

    if loadWeight:
        if if_Ori:
            currentDict = torch.load(modelSavePath + trainModelLoad)
            newDict = {}
            for key, value in currentDict.items():
                if key != "layer2.0.downsample.0.weight" and key != "layer3.0.downsample.0.weight" \
                        and key != "layer4.0.downsample.0.weight" and key != "last_linear.weight" and key != "last_linear.bias":
                    newDict[key] = value
            model.load_state_dict(newDict, strict=False)
        else:
            model.load_state_dict(torch.load(modelSavePath + trainModelLoad))

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
                        print("######")
                        print(batch_idx)
                        if truth >= 0.5:
                            targetsList.append(1)
                        else:
                            targetsList.append(0)
                        print("Val part, predict: {}, truth: {}".format(probability, truth))
                    precision, recall, _ = metrics.precision_recall_curve(y_true=targetsList, probas_pred=scoreList,
                                                                          pos_label=1)
                    aucPR = round(metrics.auc(recall, precision), 4)
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=targetsList, y_score=scoreList, pos_label=1)
                    auc = round(metrics.auc(fpr, tpr), 4)
                torch.save(model.state_dict(), modelSavePath + "Model_Oth_" + "AUC" + str(auc)
                               + "_AUCPR" + str(aucPR) + ".pth")
                model = model.train(mode=True)
        scheduler.step()
    torch.save(model.state_dict(), modelSavePath + "Model_OthF" + ".pth")





