import re
import pandas as pd
import cv2

#
# file_path = "E:\SIIM\CSVFile\ShuffleTrainData.csv"
# regex = re.compile(r"^ISIC_[0-9]{7}$")
# dataFrame = pd.read_csv(file_path)
# imageName = dataFrame["image_name"]
# filter_DF = pd.DataFrame()
#
# for i, name in enumerate(imageName):
#     if i % 1000 == 0:
#         print(i)
#     if len(regex.findall(name)) != 0:
#         filter_DF = filter_DF.append(dataFrame.iloc[i])
# print(filter_DF)
#
# filter_DF.to_csv("E:\SIIM\CSVFile\\filterTrainData.csv")

# file_path = "./CSVFile/fold_train.csv"
# data_frame = pd.read_csv(file_path)
# filter_DF = pd.DataFrame()
# for i, source in enumerate(data_frame["source"]):
#     print(i,source,sep=",")
#     if source == "SLATMD" or source == "ISIC20":
#         filter_DF = filter_DF.append(data_frame.iloc[i])
# filter_DF.to_csv("E:\SIIM\CSVFile\\TotalDataSet.csv")


# img = cv2.imread("512x512-dataset-melanoma/ISIC_0000001.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite("test.jpg",img)

### train
# file_path = "./CSVFile/TotalDataSet.csv"
# data_frame = pd.read_csv(file_path)
# imgNames = data_frame["image_id"]
# for i, name in enumerate(imgNames):
#     print(i)
#     img = cv2.imread("512x512-dataset-melanoma/" + name + ".jpg")
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("train/" + name + ".jpg", img)

### test
file_path = "./CSVFile/test.csv"
data_frame = pd.read_csv(file_path)
imgNames = data_frame["image_name"]
for i, name in enumerate(imgNames):
    print(i)
    img = cv2.imread("512x512-test/" + name + ".jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test/" + name + ".jpg", img)
