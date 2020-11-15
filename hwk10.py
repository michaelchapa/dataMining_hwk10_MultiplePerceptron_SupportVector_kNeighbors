import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
    
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
    print("Split Testing: X:", np.shape(X_test), "y:", np.shape(y_test), "\n")
    
    return X_train, X_test, y_train, y_test

###################### create_NeuralNetwork_classify #########################
# Purpose:
#   Creates a multiple level perceptron classifier to classify unseen
#   data. We create multiple classifiers with various hidden layers and 
#   activation functions. We check their accuracy and loss.
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
    # Train the classifiers
    clf_100_relu = MLPClassifier(max_iter = 10000).fit(X_train, y_train)
    clf_200_logistic = MLPClassifier(hidden_layer_sizes= 200, \
                activation = 'logistic', max_iter = 10000).fit(X_train, y_train)
        
    # Print the mean accuracy/ loss on the given test data and labels
    clf_100_relu.predict(X_test)
    clf_200_logistic.predict(X_test)
    
    clf1_loss = round(clf_100_relu.loss_, 2)
    clf1_accuracy = round(clf_100_relu.score(X_test, y_test), 2)
    
    clf2_loss = round(clf_200_logistic.loss_, 2)
    clf2_accuracy = round(clf_200_logistic.score(X_test, y_test), 2)
    
    print("Multi-layer Perceptron Classifier, " + 
          "hidden layer size: 100, activation fxn: RELU\n" +
          "Accuracy: %.2lf, Loss: %.2lf\n" % (clf1_accuracy, clf1_loss))
    print("Multi-layer Perceptron Classifier, " + 
          "hidden layer size: 200, activation fxn: Logistic\n" +
          "Accuracy: %.2lf, Loss: %.2lf" % (clf2_accuracy, clf2_loss))


def main():
    X_train, X_test, y_train, y_test = trainTestSplit_data("hwk10.csv")
    create_NeuralNetwork_classify(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()