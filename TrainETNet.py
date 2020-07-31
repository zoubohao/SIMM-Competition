import sklearn.metrics as metrics
import torch
import torchvision as tv
from torch.utils.data import DataLoader
from Models.ETNet import ETNet
from DataSet import SIMM_DataSet
import torch.nn as nn
import numpy as np
from WarmUpSch import GradualWarmupScheduler
import torch.cuda.amp as amp

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
    ## The value of alpha must less than 0.5
    ### label smooth 0.001
    alpha = 0.0
    batchSize = 4
    labelsNumber = 1
    epoch = 10
    displayTimes = 20
    reduction = 'mean'
    drop_rate = 0.2
    saveTimes = 5000
    ###
    modelSavePath = "./Model_Weight/"
    ###
    loadWeight = False
    trainModelLoad = "Model_ETNet_AUC0.8805_AUCPR0.5707.pth"
    ###
    LR = 1e-5
    warmEpoch = 1
    multiplier = 100
    ###
    device0 = "cuda:1"
    reg_lambda = 1e-5

    ### Data pre-processing
    ### Data pre-processing
    transformationTrain = tv.transforms.Compose([
        tv.transforms.RandomApply([tv.transforms.RandomRotation(degrees=25)], p=0.25),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transformationTest = tv.transforms.Compose([
        tv.transforms.Resize([576, 576]),
        tv.transforms.CenterCrop([512, 512]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    trainNegDataSet = SIMM_DataSet(root="./NegTrainAugResize",csvFile="./CSVFile/augNeg.csv",
                                   transforms=transformationTrain,train=True,alpha = alpha)
    trainPosDataSet = SIMM_DataSet(root="./PosTrainAugResize", csvFile="./CSVFile/augPos.csv",
                                   transforms=transformationTrain, train=True,alpha = alpha)
    testDataSet = SIMM_DataSet(root="./train",csvFile="./CSVFile/val.csv",
                               transforms=transformationTest,train=True, alpha = alpha)

    trainNegDataLoader = DataLoader(trainNegDataSet,batch_size=batchSize // 2,shuffle=True,pin_memory=True, num_workers=2)
    trainPosDataLoader = DataLoader(trainPosDataSet, batch_size= batchSize - batchSize // 2, shuffle=True, pin_memory=True, num_workers=2)
    testloader = DataLoader(testDataSet, batch_size=1, shuffle=False, num_workers=2)

    model = ETNet(w = 1 , d = 1.5, expand_ratio = 2, drop_ratio = drop_rate, classes_num=labelsNumber,
                  input_image_size=[352, 352]).to(device0)
    print(model)

    negLength = trainNegDataSet.__len__()
    posLength = trainPosDataSet.__len__()
    print("Positive samples number ", posLength)
    print("Negative samples number ",negLength)
    trainTimesInOneEpoch = max(negLength,posLength) // (batchSize // 2) + 1

    lossCri = nn.BCEWithLogitsLoss(reduction=reduction).to(device0)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=reg_lambda, nesterov=True)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=multiplier, total_epoch=warmEpoch,
                                       after_scheduler=cosine_scheduler)
    scaler = amp.GradScaler()
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
            labelsCuda = targetsTr.float().to(device0, non_blocking=True).view([-1,1])

            ## img, sex, age_approx, anatom
            optimizer.zero_grad()
            with amp.autocast():
                #print(imagesCuda.shape)
                predict = model(imagesCuda)
                criLoss = lossCri(predict, labelsCuda)
            scaler.scale(criLoss).backward()
            scaler.step(optimizer)
            scaler.update()
            trainingTimes += 1

            if trainingTimes % displayTimes == 0:
                with torch.no_grad():
                    print("######################")
                    print("Epoch : %d , Training time : %d" % (e, trainingTimes))
                    print("Cri Loss is %.3f " % (criLoss.item()))
                    print("Learning rate is ", optimizer.state_dict()['param_groups'][0]["lr"])
                    print("predicted labels : ", torch.sigmoid(predict[0:5]))
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
                    fpr, tpr, thresholds = metrics.roc_curve(y_true=targetsList, y_score = scoreList, pos_label=1)
                    auc = round(metrics.auc(fpr, tpr), 4)
                torch.save(model.state_dict(), modelSavePath + "Model_ETNet_" + "AUC" + str(auc)
                           + "_AUCPR" + str(aucPR) + ".pth")
                model = model.train(mode=True)
        scheduler.step()
    torch.save(model.state_dict(), modelSavePath + "Model_ETNetF_.pth")





