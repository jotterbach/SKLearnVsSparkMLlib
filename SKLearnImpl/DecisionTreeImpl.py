from sklearn import tree
import DataTransform.DataETL as etl
import numpy as np

def trainDecisionTree(data, labels):
    treeClassifier = tree.DecisionTreeClassifier()
    treeClassifier.fit(data, labels)
    return treeClassifier

def predictDecisionTree(classifier, data):
    return clf.predict(data)

def calculateVarianceScore(prediction, labels):
    return np.sqrt(np.sum(np.power(prediction-labels,2))/prediction.size)

if __name__ == "__main__":
    filePath = '/Users/jotterbach/code/SKLearnVsMLlib/RawData/train.csv'
    trainDF, testDF = etl.generateCrossValidationSet(etl.getDataAsDataframe(filePath), 0.66)
    
    data, labels = etl.extractDataAndLabels(trainDF)
    clf = trainDecisionTree(data, labels)
    
    testData, testLabels = etl.extractDataAndLabels(testDF)
    prediction = clf.predict(clf, testData)
    print calculateVarianceScore(prediction, testLabels)