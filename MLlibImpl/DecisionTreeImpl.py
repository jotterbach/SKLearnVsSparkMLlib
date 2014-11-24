from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import csv
import random as rd

def getDataAsLabelPoints(filePath):
    data = []
    with open(filePath, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        cnt = 0
        for line in reader:
            if cnt == 0:
                cnt = cnt+1
                continue
            data = data +[LabeledPoint(line[0], line[1:])]
            cnt = cnt + 1
    return data

def generateCrossValidationSet(data, ratioOfTrainingData, totalSize):
    rd.shuffle(data)
#     lengthOfDataSet = len(data)
    cut = int(totalSize*ratioOfTrainingData)
    return data[:cut], data[cut:totalSize]
    

if __name__ == "__main__":
    
    conf = SparkConf().setAppName('blub').setMaster('local')
    sc = SparkContext(conf=conf)
    
    filePath = '/Users/jotterbach/code/SKLearnVsMLlib/RawData/train.csv'
    train, test = generateCrossValidationSet(getDataAsLabelPoints(filePath), 0.66, 1000)
    print len(train), len(test)
    
    trainRDD = sc.parallelize(train)
    testRDD = sc.parallelize(test)
    
    categoricalFeaturesInfo = {}
    model = DecisionTree.trainClassifier(trainRDD, 10, categoricalFeaturesInfo, maxDepth=3)
    
    predictions = model.predict(testRDD.map(lambda x: x.features))
    labelsAndPredictions = testRDD.map(lambda lp: lp.label).zip(predictions)
    trainErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testRDD.count())
    print trainErr
    print model
    