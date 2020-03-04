import os
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
#from xgboost import XGBRegressor
from math import sqrt
import operator
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder




class NNSoftmax():
    def __init__(self):
        self.datasetPaths = np.array(["../Data/four_value_function.csv", "../Data/six_value_function.csv"])
        self.architectureFilePaths = "../Architectures/list.txt"
        self.lossFunc = 'categorical_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy', 'categorical_accuracy']
        self.validationSplit = 0.2
        self.numEphocs = np.array([256, 512])
        self.batchSize = np.array([32, 128])
        self.validationSplit = 0.2  # The parameters used in Keras in training (cross-validation)
        self.splitTrainTestPercentage = 0.9  # 1 data = 0.9 training + 0.1 test


    def prepData(self, fullFileName, scale = 1.0):
        self.df = pd.read_csv(fullFileName)
        if (scale < 1.0):
            self.df = self.df.sample(frac=scale).reset_index(drop = True)

        numOfFeatures = self.df.shape[1] - 1
        oneHot = keras.utils.to_categorical(self.df['obj'])
        for c in range(0, oneHot.shape[1]):
            self.df.insert(self.df.shape[1], str(c), oneHot[:, c], True)

        numOftraining = int(self.df.shape[0] * self.splitTrainTestPercentage)
        trainingRows = [r for r in range(0, numOftraining)]

        self.train_target_oneHot = self.df.iloc[  trainingRows, [c for c in range(numOfFeatures + 1, self.df.shape[1])] ]
        self.train_target = self.df.iloc[trainingRows, numOfFeatures]   # This refers to the column holding 'obj'
        self.train = self.df.iloc[trainingRows, [c for c in range(0, numOfFeatures)] ]

        testRows = [r for r in range(numOftraining, self.df.shape[0])]
        self.test_target_oneHot = self.df.iloc[testRows, [c for c in range(numOfFeatures + 1, self.df.shape[1])]]
        self.test_target = self.df.iloc[testRows, numOfFeatures]    # This refers to the column holding 'obj'
        self.test = self.df.iloc[testRows, [c for c in range(0, numOfFeatures)]]


        self.result = pd.DataFrame(self.test_target)
        self.scoreCollection = pd.DataFrame( {'loss': np.array([])} )
        for colName in self.metrics:
            self.scoreCollection.insert(self.scoreCollection.shape[1], colName, np.array([]))



        print('done\n')

    def buildModel(self, fullFileName):        #fullFileName refers to the architecture file
        architecture = pd.read_csv(fullFileName.strip())
        self.NN_model = Sequential()

        # The input layer :
        layerSpec = np.array(architecture.iloc[0])  # Each row of architeture defines the properties of that layer
        if (layerSpec[1] == 'Dense'):
            self.NN_model.add(
                Dense(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4],
                      input_dim=self.train.shape[1])
            )
        else:
            if (layerSpec[1] == 'Conv2D'):
                self.NN_model.add(
                    Conv2D(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4],
                        input_dim=self.train.shape[1])
                )
        # All other layers; note that the number of nodes in the last layer should not be set to what is written
        # in the architecture file. So, for the last layer, we let layerSpec[2] = number of columns used for
        # one-hot encoding.

        for i in range(0, architecture.shape[0]):
            layerSpec = architecture.iloc[i]        # Each row of architecture defines the properties of that layer

            if (i == architecture.shape[0] - 1):
                layerSpec[2] = self.test_target_oneHot.shape[1]     # to be adjusted only for the last layer.
                layerSpec[3] = 'softmax'

            if (layerSpec[1] == 'Dense'):
                self.NN_model.add(
                    Dense(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4])
                )
            else:
                if(layerSpec[1] == 'Conv2D'):
                    self.NN_model.add(
                        Conv2D(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4])
                    )

        # Compile the network :
        self.NN_model.compile(loss = self.lossFunc, optimizer = self.optimizer, metrics = self.metrics)
        self.NN_model.summary()

    def fitModel(self, nEpochs, batchSize):
        self.history = self.NN_model.fit(self.train, self.train_target_oneHot, epochs = nEpochs,
                               batch_size=batchSize, validation_split = self.validationSplit)
        self.score = self.NN_model.evaluate(self.test, self.test_target_oneHot, batch_size = batchSize)

        self.scoreCollection.loc[self.scoreCollection.shape[0]] = self.score        # adds a new record at the end of storeCollection
        print('Done with fitting the model!\n')


    def testModel(self, resulColLabel):
        test_predictions_oneHot = self.NN_model.predict(self.test)
        test_predictions = np.argmax(test_predictions_oneHot, axis = 1)

        self.result.insert(self.result.shape[1], resulColLabel, test_predictions)
        #self.result[:,self.result.shape[1]-1] = np.round(self.result[:,self.result.shape[1]-1], 2)
        print('done\n')

    def parse(self):
        file1 = open(self.architectureFilePaths, 'r')
        allArchitecPaths = file1.readlines()

        for i in range(0, self.datasetPaths.shape[0]):
            datasetPath = self.datasetPaths[i]
            modelNo = -1

            if (i == 0):
                self.prepData(datasetPath)
            else:
                self.prepData(datasetPath, 0.1)     # For larger datasets, we do not run the code for all samples.

            for nEpochs in self.numEphocs:
                for batchSize in self.batchSize:
                    for architecPath in allArchitecPaths:
                        modelNo = modelNo + 1
                        self.buildModel(architecPath)
                        self.fitModel(nEpochs, batchSize)
                        #resulColLabel = os.path.splitext(architecPath)[0] + "_n" + str(nEpochs) + "_b" + str(batchSize)
                        resulColLabel = self.getFileName(architecPath) + "_n" + str(nEpochs) + "_b" + str(batchSize)
                        # The above stores the lable of the new column that will be added to the self.result and will
                        # contain the new predictions.
                        self.testModel(resulColLabel)

            strPredictionsFileName = "../Output/SoftmaxPredict" + "_" + str(i) + ".txt"
            self.result.to_csv(strPredictionsFileName)
            #np.savetxt(strPredictionsFileName, self.result, fmt='%1.2f', delimiter=",")
            print("One dataset is done.\n")

            strScoresFileName = "../Output/linregScores" + "_" + str(i) + ".txt"
            self.scoreCollection.to_csv(strScoresFileName)

            #numOfSubArrays = np.size(self.score)
            #numOfElementsInEachSubArray = np.size(self.scoreCollection) // np.size(self.score)
            #self.scoreCollection.reshape(numOfSubArrays, numOfElementsInEachSubArray)
            #strScoresFileName = "../Output/linregScores" + "_" + str(i) + ".txt"
            #np.savetxt(strScoresFileName, self.scoreCollection, fmt='%1.3f', delimiter=",")



            print("One dataset is done.\n")
        print("Over!\n")

    def getFileName(self, strFullFileName):
        # This is an auxiliary function which gets a string, which is the full path of a filename;
        # then it extracts solely the name of the file.
        i = 0
        l = len(strFullFileName)
        while (strFullFileName[l - 1 - i] != '/'):
            i = i + 1
        return os.path.splitext(strFullFileName[l - i::1])[0]


if __name__ == "__main__":
    obj = NNSoftmax()
    #obj.prepData(obj.datasetPaths[0])
    obj.parse()



