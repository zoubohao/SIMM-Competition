import xgboost as xgb
import pandas as pd
import numpy as np



def oneHot(mapPara, oneArray):
    size = len(oneArray)
    hotSize = len(mapPara)
    zerosMatrix = np.zeros([size, hotSize],dtype=np.float32)
    for i, index in enumerate(oneArray):
        zerosMatrix[i, index] = 1.
    return zerosMatrix


def convertXGBoostData(csvFile, ageMapping, anatomMapping, sexMapping):
    data = pd.read_csv(csvFile)
    colNames = data.columns

    with open(ageMapping,"r") as rh:
        ageX = {}
        for line in rh:
            oneline = line.strip()
            if oneline != "":
                oneline = oneline.split(",")
                ageX[oneline[0]] = int(oneline[1])


    with open(anatomMapping,"r") as rh:
        anatomMap = {}
        for line in rh:
            oneline = line.strip()
            if oneline != "":
                oneline = oneline.split(",")
                anatomMap[oneline[0]] = int(oneline[1])

    with open(sexMapping,"r") as rh:
        sexMap = {}
        for line in rh:
            oneline = line.strip()
            if oneline != "":
                oneline = oneline.split(",")
                sexMap[oneline[0]] = int(oneline[1])

    ### transform age, age contains order information, so, we can't transform it into onehot encoder
    newAgeOrderInt = list()
    for ageNum in data["age_approx"]:
        newAgeOrderInt.append(ageX[str(ageNum)])

    newAnatomInt = list()
    for anaStr in data["anatom_site_general_challenge"]:
        newAnatomInt.append([anatomMap[anaStr]])


    newSexInt = list()
    for sex in data["sex"]:
        newSexInt.append([sexMap[sex]])

    ### transform one hot
    anatomOneHot = oneHot(anatomMap, newAnatomInt)
    sexOneHot = oneHot(sexMap, newSexInt)

    ### concat
    matrix = np.concatenate([np.array(newAgeOrderInt).reshape([-1,1]), anatomOneHot, sexOneHot], axis=1)

    ### concat other array
    for colName in colNames:
        if colName != "sex" and colName != "age_approx" and colName != "anatom_site_general_challenge" \
            and colName != "image_name" and colName != "target":
            matrix = np.concatenate([matrix, np.array(data[colName]).reshape([-1,1])], axis=1)
    try :
        _ = data["target"]
        return matrix, np.array(data["target"])
    except KeyError:
        print("There is no target column in names, return None.")
        return matrix, None


if __name__ == "__main__":
    testM, testT = convertXGBoostData(csvFile="./CSVFile/SecondTrainNeg.csv", ageMapping="./MappingFile/ageMapping.txt",
                                      anatomMapping="./MappingFile/anatomMapping.txt", sexMapping="./MappingFile/sexMapping.txt")
    print(testM.shape)
    print(testM[0:10])
    print(testT[0:10])

    ## shuffle the data set
    dataNeg = pd.read_csv("./CSVFile/SecondTrainNeg.csv")
    dataPos = pd.read_csv("./CSVFile/SecondTrainPos.csv")
    sColNames = dataNeg.columns
    catMap = {}
    sizeS = dataNeg.shape[0] + dataPos.shape[0]
    indexs = list(range(sizeS))
    np.random.shuffle(indexs)
    for oneName in sColNames:
        thisNegList = np.array(dataNeg[oneName]).reshape([-1,1])
        thisPosList = np.array(dataPos[oneName]).reshape([-1,1])
        catData = np.concatenate([thisNegList, thisPosList],axis=0)
        catDataShu = catData[indexs]
        catMap[oneName] = catDataShu.squeeze()
    print(catMap)
    finalData = pd.DataFrame(catMap)
    finalData.to_csv("./CSVFile/ShuffleTrainData.csv",index=False)







    










