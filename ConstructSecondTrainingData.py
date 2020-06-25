from Models.EfficientNet import  EfficientNet
from Models.ResNeSt_Ori import resnest101
import torch
import torchvision as tv
from DataSet import SIMM_DataSet
from torch.utils.data import DataLoader
import pandas as pd


def doubleNets(effi_weight_path, res_weight_path, csv_file, img_path):
    imgNames = list()
    effiResults = list()
    resResults = list()
    effi = EfficientNet.from_pretrained("efficientnet-b6", num_classes=1)
    res = resnest101(number_classes=1, drop_connect_ratio=0.25)
    effi.load_state_dict(torch.load(effi_weight_path))
    res.load_state_dict(torch.load(res_weight_path))
    effi = effi.eval()
    res = res.eval()
    ## transformations of testing
    transformations = tv.transforms.Compose([
        tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)]),
        tv.transforms.CenterCrop(size=[272, 408]),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ### note that it will return the img name of this img
    testDataSet = SIMM_DataSet(root=img_path, csvFile=csv_file,
                                   transforms=transformations, train=False)
    testDataloader = DataLoader(testDataSet,batch_size=1, shuffle=False)
    ###
    for i, (img, imgName) in enumerate(testDataloader):
        print(i)
        print(imgName)
        #print(img.shape)
        effResult = torch.sigmoid(effi(img)).squeeze()
        resResult = torch.sigmoid(res(img)).squeeze()
        imgNames.append(imgName[0])
        effiResults.append(float(effResult.detach().cpu().numpy()))
        resResults.append(float(resResult.detach().cpu().numpy()))
    return imgNames, effiResults, resResults




if __name__ == "__main__":
    effiWeight = "./Model_Weight/Model_EF0.9133417896678967.pth"
    resWeight = "./Model_Weight/Model_Re0.9265452029520296.pth"
    trainCSV = "./CSVFile/train.csv"
    imgFiles = "./train"
    trainData = pd.read_csv(trainCSV)
    sexs = trainData["sex"]
    ages = trainData["age_approx"]
    anatom = trainData["anatom_site_general_challenge"]
    targets = trainData["target"]
    imgN, effiR, resR = doubleNets(effiWeight, resWeight, trainCSV, imgFiles)
    dataframeMap = {"image_name": imgN,
                    "sex": sexs,
                    "age_approx": ages,
                    "anatom_site_general_challenge": anatom,
                    "effiResult": effiR,
                    "resResult": resR,
                    "target": targets
                    }
    dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/SecondTrainingData.csv", index=False)












