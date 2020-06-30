from Models.EfficientNet import  EfficientNet
import torch
import torchvision as tv
from DataSet import SIMM_DataSet
from torch.utils.data import DataLoader
import pandas as pd

device = "cuda:0"

def constructEffiData(effi_weight_path, csv_file, img_path, model_name):
    imgNames = list()
    effiResults = list()
    effi = EfficientNet.from_pretrained("efficientnet-" + model_name, num_classes=1).to(device)
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
    length = testDataSet.__len__()
    testDataloader = DataLoader(testDataSet,batch_size=1, shuffle=False)
    ###
    for i, (img, imgName) in enumerate(testDataloader):
        if i % 100 == 0 :
            print("Index : {}, progressing : {}".format(i, i / length + 0.))
        #print(img.shape)
        effResult = torch.sigmoid(effi(img.to(device))).squeeze()
        imgNames.append(imgName[0])
        effiResults.append(effResult.detach().cpu().numpy())
        # if i == 10:
        #     break
    return imgNames, effiResults


if __name__ == "__main__":
    #### file path
    effiWeight = "./Model_Weight/Model_EFb7_TEST_0.910.pth"   ### change here
    CSV = "./CSVFile/augNeg" \
          ".csv"
    imgFiles = "./NegTrainAugResize"
    ### name string
    modelName = "b7"
    outputName = "_TEST_0.910_Neg"  ### change here
    ### construction data or testing the result
    if_test = False
    ### construction test data

    Data = pd.read_csv(CSV)
    if if_test is False:
        sexs = Data["sex"]
        ages = Data["age_approx"]
        anatom = Data["anatom_site_general_challenge"]
        try:
            targets = Data["target"]
            imgN, effiR = constructEffiData(effiWeight, CSV, imgFiles, model_name=modelName)
            dataframeMap = {"image_name": imgN,
                            "sex": sexs,
                            "age_approx": ages,
                            "anatom_site_general_challenge": anatom,
                            "effi" + modelName + "Result": effiR,
                            "target": targets
                            }
            dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/Neg/Effi" + modelName + outputName + ".csv",
                                                          index=False)
        except KeyError:
            imgN, effiR = constructEffiData(effiWeight, CSV, imgFiles, model_name=modelName)
            dataframeMap = {"image_name": imgN,
                            "sex": sexs,
                            "age_approx": ages,
                            "anatom_site_general_challenge": anatom,
                            "effi" + modelName + "Result": effiR,
                            }
            dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/Neg/Effi" + modelName + outputName + ".csv",
                                                          index=False)

    else:
        imgN, effiR = constructEffiData(effiWeight,  CSV, imgFiles, model_name=modelName)
        dataframeMap = {
            "image_name": imgN,
            "target": effiR,
                        }
        dataframe = pd.DataFrame(dataframeMap).to_csv("./CSVFile/Effi" + modelName + outputName + ".csv", index=False)












