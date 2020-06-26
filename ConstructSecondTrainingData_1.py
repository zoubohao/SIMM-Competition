from Models.EfficientNet import  EfficientNet
import torch
import torchvision as tv
from DataSet import SIMM_DataSet
from torch.utils.data import DataLoader
import pandas as pd


def predictResultOfEffi(effi_weight_path, csv_file, img_path, model_name):
    imgNames = list()
    effiResults = list()
    effi = EfficientNet.from_pretrained("efficientnet-" + model_name, num_classes=1).to("cuda:0")
    effi.load_state_dict(torch.load(effi_weight_path))
    effi = effi.eval()
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
        #print(img.shape)
        effResult = torch.sigmoid(effi(img.to("cuda:0"))).squeeze()
        imgNames.append(imgName[0])
        effiResults.append(effResult.detach().cpu().numpy())
        # if i == 10:
        #     break
    return imgNames, effiResults




if __name__ == "__main__":
    effiWeight = "./Model_Weight/Model_EF0.9028482472324724.pth"
    CSV = "./CSVFile/test.csv"
    imgFiles = "./test"
    modelName = "b7"
    Data = pd.read_csv(CSV)
    if_test = True

    if if_test is False:
        sexs = Data["sex"]
        ages = Data["age_approx"]
        anatom = Data["anatom_site_general_challenge"]
        targets = Data["target"]
        imgN, effiR= predictResultOfEffi(effiWeight, CSV, imgFiles, model_name=modelName)
        dataframeMap = {"image_name": imgN,
                        "sex": sexs,
                        "age_approx": ages,
                        "anatom_site_general_challenge": anatom,
                        "effiResult": effiR,
                        "target": targets
                        }
        dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/SecondTraining" + modelName +"Data.csv", index=False)
    else:
        imgN, effiR = predictResultOfEffi(effiWeight,  CSV, imgFiles, model_name=modelName)
        dataframeMap = {
            "image_name": imgN,
            "target": effiR,
                        }
        dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/Effi" + modelName + "2.csv", index=False)













