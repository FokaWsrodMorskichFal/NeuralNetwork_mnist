import neural_network
import read_data
import numpy as np
import matplotlib.pyplot as plt
from ploting import plot_classification
import gzip


if __name__ == "__main__":
    np.random.seed(1)
    
    # path to directory with .gz data
    path = "./data"

    X_train, Y_train, X_test, Y_test = read_data.read_MNIST(path)

    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  '  + str(X_test.shape))
    print('Y_test:  '  + str(Y_test.shape))
    print(X_train[0:2])
    print(Y_train[0:10])

    X_train = X_train.T
    Y_train = Y_train
    X_test = X_test.T
    Y_test = Y_test

    X_train, Y_train = neural_network.data_shuffle(X_train, Y_train, True)
    #X_train, mean, std = neural_network.classification_data_normalization(X_train)
    #X_test, _, _ = neural_network.classification_data_normalization(X_test, mean, std)
    Y_train = neural_network.one_hot_encoding(Y_train)

    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  '  + str(X_test.shape))
    print('Y_test:  '  + str(Y_test.shape))

    n_classes = Y_train.shape[0]
    nn = neural_network.NeuralNetwork([784, 200, 200, n_classes], 5, 0.2, 60)
    
    #plot_classification(X_test, Y_test, n_classes)
    
    cost, parameter_progress, parameter_gradient_progress = nn.perform_training(X_train, Y_train, X_test, Y_test)

    Y_pred = nn.forward(X_test)
    Y_pred = neural_network.one_hot_decoding(Y_pred)
    print("Y_test: ", Y_test[0:50])
    print("Y_pred: ", Y_pred[0:50])

    acc=sum(Y_test==Y_pred)
    print(acc/len(Y_test))
    '''
    n_test=lean(Y_test)
    for i in range(n_test):
        if
    '''
    
    



    
    

    
