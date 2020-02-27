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


class DL():
    ## An object of this class is a DL model as a binary classifier using sigmoid function

    def __init__(self, modelName=""):
        self.NN_model = Sequential()
        self.modelName = modelName  # This is used as the name of the file containing the model info (saved after training)
        self.lossFunc = 'binary_crossentropy'
        self.optimizer = 'adam'
        self.metrics = ['accuracy']
        self.validationSplit = 0.2  # The parameters used in Keras in training (cross-validation)

    def read_data(self, trainData):
        self.numOfFeatures = trainData.shape[1] - 1
        self.train_target = trainData['obj']
        self.train = trainData.drop(['obj'], axis=1)

    def saveTofile(self, filename):  # Recieves a .h5 filename here.
        self.NN_model.save(filename)

    def loadFromFile(self, filename):
        self.NN_model = keras.models.load_model(filename)

    def createArchitecture(self, filename=""):
        if (filename == ""):
            # This is the default architecture.
            self.architecture = pd.DataFrame({'numNodes': np.array([16, 128, 128, 128, 128, 1]),
                                              'kernel_initializer': np.array(
                                                  ['normal', 'normal', 'normal', 'normal', 'normal', 'normal']),
                                              'activation': np.array(
                                                  ['relu', 'relu', 'relu', 'relu', 'relu', 'sigmoid'])})
            self.lossFunc = 'binary_crossentropy'
            self.optimizer = 'adam'
            self.metrics = ['accuracy']
            self.numEphocs = 50
            self.batchSize = 32
            self.validationSplit = 0.2

        else:
            architecture = pd.read_csv(filename.strip())

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
                layerSpec = architecture.iloc[i]  # Each row of architeture defines the properties of that layer
                if (layerSpec[1] == 'Dense'):
                    self.NN_model.add(
                        Dense(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4])
                    )
                else:
                    if (layerSpec[1] == 'Conv2D'):
                        self.NN_model.add(
                            Conv2D(layerSpec[2], activation=layerSpec[3], kernel_initializer=layerSpec[4])
                        )
        print("NN_model is ready.\n")

    def trainModel(self, nEpochs, batchSize):
        # Compile the network :
        self.NN_model.compile(loss=self.lossFunc, optimizer=self.optimizer, metrics=self.metrics)
        self.NN_model.summary()

        # Actual training of the model...
        self.NN_model.fit(self.train, self.train_target,
                       epochs=nEpochs,
                       batch_size=batchSize,
                       validation_split=self.validationSplit)

        ############################# Ad-hoc test ######################################
        ###sampleTest = {'b1': [2, 2, 7, 9],
        ###              'b2': [9, 8, 3, 3],
        ###              'b3': [6, 9, 1, 1],
        ###             'b4': [2, 1, 1, 1],
        ###              'obj': [1, 1, 0, 0]}
        ###sampleTest = pd.DataFrame(sampleTest)
        ###sampleTest.drop(['obj'], axis=1, inplace=True)
        ###p = self.NN.predict(sampleTest).flatten()
        ################################################################################


class NNOrdianlBinary:
    def __init__(self):
        self.datasetPaths = np.array(["../Data/four_value_function.csv", "../Data/six_value_function.csv"])
        self.architectureFilePaths = "../Architectures/list.txt"
        self.numEphocs = np.array([50, 150])
        self.batchSize = np.array([32, 128])
        self.splitTrainTestPercentage = 0.9  # 1 data = 0.9 training + 0.1 test

        self.DLs = []       # This stores the collection of DL models

    def prepData(self, fullFileName, scale=1.0):
        self.df = pd.read_csv(fullFileName)
        if (scale < 1.0):
            self.df = self.df.sample(frac=scale).reset_index(drop = True)

        numOftraining = int(self.df.shape[0] * self.splitTrainTestPercentage)
        self.train = self.df[:numOftraining]
        self.train = self.train.sort_values(by=['obj'])
        trainObjectiveClasses = self.train['obj'].value_counts().to_dict()  # stroes the classes (objectie values) and their frequency
        # -----------------------------------------------------------
        # The following constructs the list of pairs (a, b), where
        # a is an obj value appeared in the training set, and b is
        # the number of times that the value a has appeared.
        # -----------------------------------------------------------
        self.trainObjectiveClassesSorted = []
        for key in sorted(trainObjectiveClasses.keys()):
            self.trainObjectiveClassesSorted.append((key, trainObjectiveClasses[key]))


        #self.train.drop(['obj'], axis=1, inplace=True)
        self.numOfClasses = len(trainObjectiveClasses.keys())

        self.test = self.df[numOftraining:]
        self.test_target = self.test['obj']  # self.test will be test X
        self.test.drop(['obj'], axis=1, inplace=True)  # self.test will be test Y

        self.result = np.ndarray(shape=(self.test_target.shape[0], 1), dtype=float)
        self.result = np.insert(self.result, 1, np.round(self.test_target, 2), axis=1)
        self.result = np.delete(self.result, 0, axis=1)
        print('test_target is added to the self.result.\n')

    def buildAndTrainModels(self, architecPath="", nEpochs=0, batchSize=0, fileNamePattern=""):
        if (fileNamePattern == ""):
            if (architecPath == ""):
                print("Either achitecture or the former DF files must be provided.\n")
                quit()

            else:
                for m in range(0, self.numOfClasses - 1):
                    dl = DL()
                    dlFileName = "DL_" + str(m) + ".h5"

                    # -----------------------------------------------------------------------------
                    # For each model m, we need to construct a customized binary training dataset
                    # (from the original training data) in which the labels of all observations
                    # with label <= classType(m) are set to zero, and the rest of the labels are
                    # set to one.
                    # -----------------------------------------------------------


                    binaryTrainData = self.train
                    binaryTrainData = binaryTrainData.rename(columns={'obj': 'objValue'})

                    numZeroLabels = 0
                    for i in range(0, m + 1):
                        numZeroLabels = numZeroLabels + self.trainObjectiveClassesSorted[i][1]
                    numOneLabels = binaryTrainData.shape[0] - numZeroLabels

                    binaryTrainData['obj'] = np.concatenate(
                        (np.zeros((numZeroLabels, 1), dtype=int), np.ones((numOneLabels, 1), dtype=int)), axis=0)
                    binaryTrainData = binaryTrainData.drop(['objValue'], axis=1)

                    # ---------------------------------------------------------------
                    # The binary training dataset is now used to train the data.
                    dl.read_data(binaryTrainData)
                    dl.createArchitecture(architecPath)
                    dl.trainModel(nEpochs,batchSize)
                    # ---------------------------------------------------------------

                    self.DLs.append(dl)
                    dl.saveTofile(dlFileName)
                    print("\n------------------------\nModel " + str(m) + " is created.\n-------------------\n")

        else:
            if (fileNamePattern == "DL_"):
                for m in range(0, self.numOfClasses - 1):
                    fileName = fileNamePattern + str(m) + ".h5"
                    dl = DL(fileName)
                    dl.loadFromFile(fileName)
                    self.DLs.append(dl)
                    print("\n------------------------\nModel " + str(m) + " is loaded.\n-------------------\n")

            else:
                print("Not a valid name for the DL files.\n")

    def evaluate(self):
        print("\n------------------------\nThere are " + str(len(self.DLs)) + " classification models.\n------------------------\n")

        # --------------------------------------------------------------------
        # We first create probMatrix with columns corresponding to
        # 1 - f_0(x)=1 - P{X>=c_1},
        # P{X>c_1} - P{X>c_2},
        # P{X>c_2} - P{X>c_3},
        # ...
        # {X>c_q (which is the label for the one but last type of labels)}.
        # Accordingly, probMatrix has numClasses number of columns.
        # --------------------------------------------------------------------

        self.probMatrix = np.zeros(self.test.shape[0])
        previousClassPrediction = np.array([], dtype=float)
        for m in range(0, len(self.DLs)):
            dl = self.DLs[m]
            #binaryPrediction = dl.NN_model.predict(self.test.drop(['obj'], axis=1))
            binaryPrediction = dl.NN_model.predict(self.test).flatten()
            if (m == 0):
                self.probMatrix = 1 - binaryPrediction
                previousClassPrediction = binaryPrediction
            else:
                self.probMatrix = np.column_stack((self.probMatrix, previousClassPrediction - binaryPrediction))
                previousClassPrediction = binaryPrediction


        predictedLabels = np.zeros((self.test.shape[0], 1), dtype=int)
        for i in range(0, self.test.shape[0]):
            arr1D = self.probMatrix[i, :]
            idx = np.where(arr1D == np.amax(arr1D))

            predictedLabels[i][0] = self.trainObjectiveClassesSorted[idx[0][0]][0]


        self.result = np.column_stack((self.result, predictedLabels))
        self.DLs=[]         # At this time, we discard all the models in the DLs.
        print("Stop here.\n")

    def parse(self):
        file1 = open(self.architectureFilePaths, 'r')
        allArchitecPaths = file1.readlines()
        for i in range(0, self.datasetPaths.shape[0]):
            datasetPath = self.datasetPaths[i]
            modelNo = -1
            memo = [' ' for m in range(0, self.datasetPaths.size * self.numEphocs.size * np.size(allArchitecPaths))]
            # this will be used to store the memo indicating the info in each column of self.result
            self.scoreCollection = np.empty(shape=(0, 0))

            if (i == 0):
                self.prepData(datasetPath)
            else:
                self.prepData(datasetPath, 0.1)

            for nEpochs in self.numEphocs:
                for batchSize in self.batchSize:
                    for architecPath in allArchitecPaths:
                        modelNo = modelNo + 1
                        self.buildAndTrainModels(architecPath,nEpochs,batchSize)
                        self.evaluate()

            strPredictionsFileName = "../Output/ordinalBinaryPredict" + "_" + str(i) + ".txt"
            np.savetxt(strPredictionsFileName, self.result, fmt='%1.2f', delimiter=",")
            print("One dataset is done.\n")

            numOfSubArrays = np.size(self.score)
            numOfElementsInEachSubArray = np.size(self.scoreCollection) // np.size(self.score)
            self.scoreCollection.reshape(numOfSubArrays, numOfElementsInEachSubArray)
            strScoresFileName = "../Output/ordinalBinaryScores" + "_" + str(i) + ".txt"
            np.savetxt(strScoresFileName, self.scoreCollection, fmt='%1.3f', delimiter=",")

            strMemoFileName = "../Output/ordinalBinaryMemo" + "_" + str(i) + ".txt"
            np.savetxt(strMemoFileName, memo, delimiter=";", fmt="%s")
            print("One dataset is done.\n")


def main():
    obj = NNOrdianlBinary()
    obj.parse()

if __name__ == "__main__":
    main()



