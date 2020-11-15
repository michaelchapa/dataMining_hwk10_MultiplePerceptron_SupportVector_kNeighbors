import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

############################# generateData ###################################
# Purpose:
#   Creates 20 data tuples based on csv data, newData is scaled to original
#   data dimensionality and values.
# Parameters:
#   csvFile    csv file    n-dimensional features and class label
# Returns:
#   newData    numpyArray  generated data with n-feature dimensionality. 
#                          Doesn't return class label for each instance
# Notes:
#   Honestly, quite a useless function.
def generateData(csvFile):
    data = pd.read_csv(csvFile).to_numpy()
    X, y = data[:, 0:-1], data[:, -1]
    
    scaler = MinMaxScaler()
    X_new = scaler.fit_transform(X, y)
    
    return X_new[0:20, :]
    
########################## trainTestSplit_data ###############################
# Purpose:
#   splits the data into Training and Testing tuples, Testing tuples are 
#   specified to only have 20 instances while Training is all other instances.
# Parameters:
#   csvFile     csvFile     n-dimensional features and class label
# Returns:
#   X_train     Array       Training attributes, proportional to y_train
#   X_test      Array       Testing attributes, proportional to y_test
#   y_train     Array       Training class labels, proportional to X_train
#   y_test      Array       Testing class labels, proportional to X_test
# Notes:
#   None
def trainTestSplit_data(csvFile):
    data = pd.read_csv(csvFile)
    data = data.to_numpy()
    X_train, X_test, y_train, y_test = \
        train_test_split(data[:, 0:-1], data[:, -1], test_size = 20)
        
    print("Split Training: X:", np.shape(X_train), "y:", np.shape(y_train))
    print("Split Testing: X:", np.shape(X_test), "y:", np.shape(y_test))
    
    return X_train, X_test, y_train, y_test

###################### create_NeuralNetwork_classify #########################
# Purpose:
#   Creates a multiple level perceptron classifier to classify several unseen
#   data. We create multiple classifiers with various hidden layers and 
#   activation functions. We check their accuracy.
# Parameters:
#   X_train     Array       Training attributes, proportional to y_train
#   X_test      Array       Testing attributes, proportional to y_test
#   y_train     Array       Training class labels, proportional to X_train
#   y_test      Array       Testing class labels, proportional to X_test
# Returns:
#   None
# Notes:
#   None
def create_NeuralNetwork_classify(X_train, X_test, y_train, y_test):
    print("Hi :P")


def main():
    X_train, X_test, y_train, y_test = trainTestSplit_data("hwk10.csv")
    

if __name__ == "__main__":
    main()