import numpy as np
import csv
import gzip


def read_data(file_name, classification=False):
    with open(file_name, mode='r') as file:
        csv_reader = csv.reader(file)
        csv_reader.__next__()
        data = np.array(list(csv_reader), dtype=np.float32)
    if classification:
        return data.T[0:2].T, data.T[2].T
    return data.T[0:1].T, data.T[1:2].T


def read_MNIST(path):
    file_names = ['train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz',
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz']

    image_size = 28
    # up to 60 000
    num_images_train = 20000
    # up to 10 000
    num_images_test = 1000

    f = gzip.open(path + '/' + file_names[0],'r')

    f.read(16)
    buf = f.read(image_size * image_size * num_images_train)
    X_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X_train = X_train.reshape(num_images_train, image_size, image_size, 1)
    X_train=X_train.reshape(X_train.shape[0], -1)

    f = gzip.open(path + '/' + file_names[1],'r')
    f.read(8)

    Y_train = [0 for i in range(num_images_train)]
    for i in range(num_images_train):
        buf = f.read(1)
        Y_train[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    Y_train=np.array(Y_train).reshape(num_images_train)

    f = gzip.open(path + '/' + file_names[2],'r')
    f.read(16)

    buf = f.read(image_size * image_size * num_images_test)
    X_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    X_test = X_test.reshape(num_images_test, image_size, image_size, 1)
    X_test=X_test.reshape(X_test.shape[0], -1)


    f = gzip.open(path + '/' + file_names[3],'r')
    f.read(8)

    Y_test = [0 for i in range(num_images_test)]
    for i in range(num_images_test):
        buf = f.read(1)
        Y_test[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    Y_test=np.array(Y_test).reshape(num_images_test)

    return X_train, Y_train, X_test, Y_test