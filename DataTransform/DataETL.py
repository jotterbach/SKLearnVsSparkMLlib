import pandas as pd
import numpy as np

def getDataAsDataframe(filePath):
    df = pd.DataFrame.from_csv(filePath)
    df['index'] = range(df.shape[0])
    df['label'] = df.index
    df = df.set_index('index')
    return df

def getLabels(dataframe):
    return dataframe['label']

def generateCrossValidationSet(dataframe, ratioOfTrainingData):
    df = dataframe.reindex(np.random.permutation(dataframe.index)).sort()
    lengthDf = df.shape[0]
    cut = int(lengthDf*ratioOfTrainingData)
    return df[:cut], df[cut:]

def extractDataAndLabels(dataframe):
    labels = getLabels(dataframe)
    data = dataframe.drop('label',1)
    return data.as_matrix(), labels.as_matrix()

if __name__=="__main__":
    filePath = '/Users/jotterbach/code/SKLearnVsMLlib/RawData/train.csv'
    TrainingDF = getDataAsDataframe(filePath)
    print TrainingDF.keys()[-5:]
    train, test = generateCrossValidationSet(TrainingDF, 0.66)
    print train.shape, test.shape
    print extractDataAndLabels(TrainingDF)
    
    
    

