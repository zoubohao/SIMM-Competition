import xgboost as xgb
from ConvertDataFormatXGBoost import convertXGBoostData
import pandas as pd

if __name__ == "__main__":
    ## config
    trainCSVFile = "./CSVFile/ShuffleTrainData.csv"
    valCSVFile = "./CSVFile/SecondVal.csv"
    testCSVFile = "./CSVFile/SecondTest.csv"

    ### import data
    trainData, trainLabels = convertXGBoostData(csvFile=trainCSVFile, ageMapping="./MappingFile/ageMapping.txt",
                                      anatomMapping="./MappingFile/anatomMapping.txt", sexMapping="./MappingFile/sexMapping.txt")
    valData, valLabels = convertXGBoostData(csvFile=valCSVFile, ageMapping="./MappingFile/ageMapping.txt",
                                      anatomMapping="./MappingFile/anatomMapping.txt", sexMapping="./MappingFile/sexMapping.txt")
    testData , _ = convertXGBoostData(csvFile=testCSVFile, ageMapping="./MappingFile/ageMapping.txt",
                                      anatomMapping="./MappingFile/anatomMapping.txt", sexMapping="./MappingFile/sexMapping.txt")

    dTrain = xgb.DMatrix(trainData, label=trainLabels)
    dVal = xgb.DMatrix(valData,label=valLabels)
    dTest = xgb.DMatrix(testData)

    print("The columns of D-Matrix is {}".format(dTrain.num_col()))
    print("The rows of D-Matrix is {}".format(dTrain.num_row()))
    ### parameter setting
    num_round = 1000
    param = {'max_depth': 6,  ## Increasing this value will make the model more complex and more likely to overfit.
             'eta': 0.033,  ## Step size shrinkage used in update to prevents overfitting. Learning rate
             "gamma": 0.044,   ## The larger gamma is, the more conservative the algorithm will be
             "min_child_weight": 6.7,  ## The larger min_child_weight is, the more conservative the algorithm will be.
             "lambda": 2.7,  ### L2 regu for weights.
             'objective': 'binary:logistic',
             'eval_metric': ['auc', 'aucpr']}
    evallist = [(dVal, 'eval'), (dTrain, 'train')]

    ### train
    bst = xgb.train(param, dTrain, num_round, evallist,early_stopping_rounds=6)
    ypred = bst.predict(dTest)
    #print(ypred)

    testDF = pd.read_csv(testCSVFile)
    imgNames = testDF["image_name"]
    resultMap = {"image_name": imgNames,
                 "target": ypred}
    pd.DataFrame(resultMap).to_csv("./CSVFile/XGBoostR.csv",index=False)

    ### save
    bst.save_model('./Model_Weight/xgBoost.model')

    # ##
    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('./Model_Weight/xgBoostBaseline.model')  # load data

