from DataSet import SIMM_DataSet
import torchvision as tv
from torch.utils.data import DataLoader
from Models.MobileNetV2 import MobileNetV2
import torch
import pandas as pd
import numpy as np



model_weight = "./Model_Weight/Model_MB0.9772727272727273.pth"
device0 = "cuda:0"

transformationTest = tv.transforms.Compose([
    tv.transforms.Resize([512, 512]),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.8060, 0.6204, 0.5902], [0.0823, 0.0963, 0.1085]),
])


testDataSet = SIMM_DataSet(root="./test" ,csvFile="./test.csv" ,transforms=transformationTest ,train=False)
testDataLoader = DataLoader(testDataSet ,batch_size=1 ,shuffle=False)

### Evaluation mode
model = MobileNetV2(3,2,0.2, w=2, d=2).to(device0)
model.load_state_dict(torch.load(model_weight))
model = model.eval()

namesList = list()
targetsList = list()

for i ,(imgs, sexs, ages, anatoms, names) in enumerate(testDataLoader):
    imgsTeCuda = imgs.to(device0, non_blocking=True)
    sexsTeCuda = sexs.to(device0, non_blocking=True)
    agesTeCuda = ages.to(device0, non_blocking=True)
    anatomsTeCuda = anatoms.to(device0, non_blocking=True)

    ## img, sex, age_approx, anatom
    outputs = model(imgsTeCuda, sexsTeCuda, agesTeCuda, anatomsTeCuda)
    _ , predicted = outputs.max(1)

    namesList.append(names[0])
    target = predicted.detach().cpu().numpy().squeeze()
    targetsList.append(target)

    if i % 100 == 0:
        print(i)
        print(names[0] + ":" + str(target))



result = {"image_name": namesList, "target" : targetsList}
dataFrame = pd.DataFrame(result)
dataFrame.to_csv("./Result.csv",index=False)






