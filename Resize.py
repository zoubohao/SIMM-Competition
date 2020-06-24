from PIL import Image
import pandas as pd
import os
import numpy as np
import torchvision as tv

dataInfor = pd.read_csv("./CSVFile/augPos.csv")
openPath = "./PosTrainAug"
savePath = "./PosTrainAugResize"

imgsNames = np.array(dataInfor["image_name"])


re = tv.transforms.Resize([int(272 * 1.118), int(408 * 1.118)])
for i,name in enumerate(imgsNames):
    print(i)
    img = Image.open(os.path.join(openPath, name + ".jpg")).convert("RGB")
    reImg = re(img)
    reImg.save(os.path.join(savePath, name + ".jpg"))







