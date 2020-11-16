import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
    
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

###################### create_supportVector_classify #########################
# Purpose:
#   Creates multiple C-Support Vector classifiers with various parameters.
#   We're observing the different Accuracies.
# Parameters:
#   X_train     Array       Training attributes, proportional to y_train
#   X_test      Array       Testing attributes, proportional to y_test
#   y_train     Array       Training class labels, proportional to X_train
#   y_test      Array       Testing class labels, proportional to X_test
# Returns:
#   None
# Notes:
#   None
def create_supportVector_classify(X_train, X_test, y_train, y_test):
    clf_rbf = SVC(C = .1, gamma = "auto").fit(X_train, y_train)
    clf_rbf.predict(X_test)
    clf_rbf_accuracy = clf_rbf.score(X_test, y_test)
    
    clf_poly = SVC(C = .001, gamma = "auto", kernel = "poly").fit(X_train, y_train)
    clf_poly.predict(X_test)
    clf_poly_accuracy = clf_poly.score(X_test, y_test)
    
    clf_sig = SVC(C = 1, gamma = "auto", kernel = "sigmoid").fit(X_train, y_train)
    clf_sig.predict(X_test)
    clf_sig_accuracy = clf_sig.score(X_test, y_test)
    
    print("C-Support Vector Classifier, " + 
          "C: .1, kernel: rbf\n" +
          "Accuracy: %.2lf\n" % (clf_rbf_accuracy))
    print("C-Support Vector Classifier, " + 
          "C: .001, kernel: poly\n" +
          "Accuracy: %.2lf\n" % (clf_poly_accuracy))
    print("C-Support Vector Classifier, " + 
          "C: 1, kernel: sigmoid\n" +
          "Accuracy: %.2lf\n" % (clf_sig_accuracy))
    
#################### create_kNeighbors_classify ##############################
# Purpose:
#   None
# Parameters:
#   None
# Returns:
#   None
# Notes:
#   None
def create_kNeighbors_classify(X_train, X_test, y_train, y_test):
    neigh2 = KNeighborsClassifier(n_neighbors = 2)
    neigh2.fit(X_train, y_train)
    neigh2.predict(X_test)
    kNeigh_2_acc = neigh2.score(X_test, y_test)
    
    neigh3 = KNeighborsClassifier(n_neighbors = 3)
    neigh3.fit(X_train, y_train)
    neigh3.predict(X_test)
    kNeigh_3_acc = neigh3.score(X_test, y_test)
    
    neigh4 = KNeighborsClassifier(n_neighbors = 4)
    neigh4.fit(X_train, y_train)
    neigh4.predict(X_test)
    kNeigh_4_acc = neigh4.score(X_test, y_test)
    
    print("K-nearest neighbors Classifier, " + 
          "n_neighbors: 2\n" +
          "Accuracy: %.2lf\n" % (kNeigh_2_acc))
    print("K-nearest neighbors Classifier, " + 
          "n_neighbors: 3\n" +
          "Accuracy: %.2lf\n" % (kNeigh_3_acc))
    print("K-nearest neighbors Classifier, " + 
          "n_neighbors: 4\n" +
          "Accuracy: %.2lf\n" % (kNeigh_4_acc))
    

def main():
    X_train, X_test, y_train, y_test = trainTestSplit_data("hwk10.csv")
    create_NeuralNetwork_classify(X_train, X_test, y_train, y_test)
    create_supportVector_classify(X_train, X_test, y_train, y_test)
    create_kNeighbors_classify(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()