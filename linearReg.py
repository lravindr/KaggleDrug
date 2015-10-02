__author__ = 'lokesh'

#Submission for Rossman Sales Store

#Idea is to treat predict sales of each store individually.
#Construct a linear regression model for each store and predict the sales of the store for the test data.

import pandas as pd
import numpy as np
from sklearn import linear_model

# Function to predict the sales volume per store
def predictPerStore(perStoreTrainData, perStoreTestData, i):
    perStoreNumpyTrainData = perStoreTrainData.as_matrix(
        columns=['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday', 'Sales'])
    perStoreNumpyTestData = perStoreTestData.as_matrix(columns=['DayOfWeek', 'Open', 'Promo', 'SchoolHoliday'])

    trainData = perStoreNumpyTrainData
    targetTrain = trainData[:, 4]
    inputTrain = trainData[:, :3]

    testData = perStoreNumpyTestData
    inputTest = testData[:, :3]

    #Using a simple regression model to train
    regr = linear_model.LinearRegression()
    regr.fit(inputTrain, targetTrain)

    #Use the model to do predictions
    predictions = np.int32(np.round(regr.predict(inputTest)))
    predictions[predictions < 0] = 0

    perStoreTestData['Sales'] = pd.Series(predictions, index=perStoreTestData.index)
    return perStoreTestData


#Read all the csv files with the 'StateHoliday' resolution
trainData = pd.read_csv('train.csv', dtype={'StateHoliday': 'string'})  # Resolve ambiguity in the data type
storeData = pd.read_csv('store.csv')
testData = pd.read_csv('test.csv')

#Find the list of stores for which sales has to be predicted.
#All the stores in the train data don't exist in the test data.
stores = testData.Store.unique()

#Predict per store and append to a dataframe
predictedData = pd.DataFrame()
for i in stores:
    perStorePredictions = predictPerStore(trainData[trainData['Store'] == i], testData[testData['Store'] == i].fillna(1), i)
    #Dataframe append is not in-place.
    predictedData = predictedData.append(perStorePredictions)

#Sort according to ID and write to a csv file
predictedData.sort('Id').to_csv('predicted_perstoreregression.csv', columns=["Id","Sales"], index=False)