import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
from xgboost import XGBRegressor
from math import sqrt
import operator

class NNRegress():
    def __init__(self):
        self.datasetPaths = np.array(["../Data/four_value_function.csv", "../Data/six_value_function.csv"])
        self.architectureFilePaths = "../Architectures/list.txt"
        self.lossFunc = 'mse'
        self.optimizer = 'adam'
        self.metrics = ['mse', 'mae']
        self.validationSplit = 0.2
        self.numEphocs = np.array([50,150])
        self.batchSize = np.array([32, 128])
        self.validationSplit = 0.2  # The parameters used in Keras in training (cross-validation)
        self.splitTrainTestPercentage = 0.9  # 1 data = 0.9 training + 0.1 test

    def prepData(self, fullFileName, scale = 1.0):
        self.df = pd.read_csv(fullFileName)
        if (scale < 1.0):
            self.df = self.df.sample(frac=scale).reset_index(drop = True)

        numOftraining = int(self.df.shape[0] * self.splitTrainTestPercentage)
        self.train = self.df[:numOftraining]
        self.train_target = self.train['obj']  # self.train_target will be training Y
        self.train.drop(['obj'], axis=1, inplace=True)  # self.train will be training X

        self.test = self.df[numOftraining:]

        self.test_target = self.test['obj']  # self.test will be test X
        self.test.drop(['obj'], axis=1, inplace=True)  # self.test will be test Y
        self.result = np.ndarray( shape=(self.test_target.shape[0],1), dtype = float)
        self.result = np.insert(self.result, 1, np.round(self.test_target,2), axis = 1)
        self.result = np.delete(self.result, 0, axis = 1)
        print('done\n')
        #self.result = np.insert(self.result, 1, self.test_target, axis=1)
        #self.result = np.insert(self.result, 1, np.zeros(self.result.shape[0]), axis=1)
        #print('done twice!\n')


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
        # All other layers
        for i in range(0, architecture.shape[0]):
            layerSpec = architecture.iloc[i]        # Each row of architeture defines the properties of that layer
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
        self.history = self.NN_model.fit(self.train, self.train_target, epochs = nEpochs,
                               batch_size=batchSize, validation_split = self.validationSplit)
        self.score = self.NN_model.evaluate(self.test, self.test_target, batch_size = batchSize)

        self.scoreCollection= np.append(self.scoreCollection, self.score)
        #else:
        #    self.scoreCollection = np.insert(self.scoreCollection, self.scoreCollection.shape[1], self.score, axis = 1)


    def testModel(self):
        test_predictions = self.NN_model.predict(self.test).flatten()
        self.result = np.insert(self.result, self.result.shape[1], test_predictions, axis=1)
        self.result[:,self.result.shape[1]-1] = np.round(self.result[:,self.result.shape[1]-1], 2)
        print('done\n')



    def parse(self):
        file1 = open(self.architectureFilePaths, 'r')
        allArchitecPaths = file1.readlines()

        for i in range(0, self.datasetPaths.shape[0]):
            datasetPath = self.datasetPaths[i]
            modelNo = -1
            memo = [' ' for m in range(0, self.datasetPaths.size * self.numEphocs.size * np.size(allArchitecPaths)) ]
                    # this will be used to store the memo indicating the info in each column of self.result
            self.scoreCollection = np.empty(shape=(0,0))#.array([], dtype=float) # This will store vectors of scores (which are the output of evaluate function in Keras)

            if (i == 0):
                self.prepData(datasetPath)
            else:
                self.prepData(datasetPath, 0.1)

            for nEpochs in self.numEphocs:
                for batchSize in self.batchSize:
                    for architecPath in allArchitecPaths:
                        modelNo = modelNo + 1
                        self.buildModel(architecPath)
                        self.fitModel(nEpochs, batchSize)
                        self.testModel()
                        str1 = "architecture, " + architecPath.strip() + ", nEpochs, " + str(nEpochs) + ", batch_size, ", str(batchSize)
                        memo[modelNo] = str1

            strPredictionsFileName = "../Output/linregPredict" + "_" + str(i) + ".txt"
            np.savetxt(strPredictionsFileName, self.result, fmt='%1.2f', delimiter=",")
            print("One dataset is done.\n")

            numOfSubArrays = np.size(self.score)
            numOfElementsInEachSubArray = np.size(self.scoreCollection) // np.size(self.score)
            self.scoreCollection.reshape(numOfSubArrays, numOfElementsInEachSubArray)
            strScoresFileName = "../Output/linregScores" + "_" + str(i) + ".txt"
            np.savetxt(strScoresFileName, self.scoreCollection, fmt='%1.3f', delimiter=",")

            strMemoFileName = "../Output/linregMemo" + "_" + str(i) + ".txt"
            np.savetxt(strMemoFileName, memo, delimiter=";", fmt="%s")
            print("One dataset is done.\n")

            print("One dataset is done.\n")
        print("Over!\n")

def main():
    obj = NNRegress()
    obj.parse()

if __name__ == "__main__":
    main()






    # sr = SAMPLEREGRESS()
    # sr.run()
