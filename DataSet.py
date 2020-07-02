from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image
import torch
import torchvision as tv
import numpy as np


def convertStringToFloat(vec,mappingFile):
    sToF_map = {}
    transVec = []
    with open(mappingFile,"r") as rh:
        for line in rh:
            oneLine = line.strip("\n").split(",")
            sToF_map[oneLine[0]] = float(oneLine[1])
    for s in vec:
        transVec.append(sToF_map[str(s)])
    return transVec


class SIMM_DataSet(Dataset):

    def __init__(self,root,csvFile,transforms = None,train = True, alpha = 0.1):
        super().__init__()
        self.root = root
        dataInfor = pd.read_csv(csvFile)
        self.imgs = list(dataInfor["image_name"])
        if train:
            labels = np.array(dataInfor["target"], dtype=np.float32)
            self.labels = np.where(labels == 1, 1 - alpha, alpha)
        self.if_train = train
        self.transforms = transforms


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        if self.if_train:
            target = torch.as_tensor(self.labels[idx]).float()

            return img, target
        else:

            return img, self.imgs[idx]


    def __len__(self):
        return len(self.imgs)





if __name__ == "__main__":
    # unique sex length  3
    # unique age_approx length  19
    # unique anatoms length  7
    # import numpy as np
    # data = pd.read_csv("./train.csv")
    # dataTest = pd.read_csv("./test.csv")
    # data = pd.concat([data,dataTest],axis=0)
    # naCheck = np.array(pd.isna(data).values, dtype=np.float32).sum()
    # print(naCheck)
    # print(data.columns)
    # sexs = data["sex"]
    # ages = data["age_approx"]
    # anatoms = data["anatom_site_general_challenge"]
    # print("unique sex length ", len(np.unique(sexs)))
    # print("unique age_approx length ", len(np.unique(ages)))
    # print("unique anatoms length ",len(np.unique(anatoms)))
    #
    # with open("./ageMapping.txt","w") as wh:
    #     k = 0
    #     ages = np.arange(start=0,stop = 185,step = 5)
    #     for i , age in enumerate(ages):
    #         wh.write(str(age) + "," + str(i) + "\n")
    #     wh.write("-1,37")
    #
    # with open("./sexMapping.txt","w") as wh:
    #     for i , se in enumerate(np.unique(sexs)):
    #         wh.write(se + "," + str(i) + "\n")
    #
    # with open("./anatomMapping.txt","w") as wh:
    #     for i , a in enumerate(np.unique(anatoms)):
    #         wh.write(a + "," + str(i) + "\n")

    transformationTrain = tv.transforms.Compose([
        tv.transforms.Resize([512,512]),
        tv.transforms.ToTensor(),
    ])
    testDataSet = SIMM_DataSet(root="./train",csvFile="./CSVFile/train.csv",transforms=transformationTrain,train=True)
    testDataLoader = DataLoader(testDataSet,batch_size=1,shuffle=False)
    for i ,(imgs, targ) in enumerate(testDataLoader):
        print(imgs.shape)
        print(targ)

    # from Tools import get_mean_and_std
    # ## tensor([0.8060, 0.6204, 0.5902]), tensor([0.0823, 0.0963, 0.1085])
    # print(get_mean_and_std(testDataSet))












