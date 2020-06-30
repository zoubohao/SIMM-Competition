import xgboost as xgb
from ConvertDataFormatXGBoost import convertXGBoostData




if __name__ == "__main__":
    ## config
    trainCSVFile = ""
    valCSVFile = ""

    ### import data 
    data, labels = convertXGBoostData(csvFile=trainCSVFile, ageMapping="./MappingFile/ageMapping.txt",
                                      anatomMapping="./MappingFile/anatomMapping.txt", sexMapping="./MappingFile/sexMapping.txt")
    print(data[0:10])
    print(labels[0:10])

    size = data.shape[0]



    #dtrain = xgb.DMatrix(trainData, label=trainLabel)
    param = {'max_depth': 50, 'eta': 1, 'objective': 'binary:logistic', 'nthread': 4, 'eval_metric': 'auc'}





