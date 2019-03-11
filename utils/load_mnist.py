from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import random
import numpy as np

def load_mnist_17(train_size = 30, test_size = 2000):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    train_17s = np.logical_or(y_train == 1, y_train == 7)
    test_17s = np.logical_or(y_test == 1, y_test == 7)
    x_train = x_train[train_17s]
    y_train = y_train[train_17s] 
    x_test = x_test[test_17s]
    y_test = y_test[test_17s]

    y_train = (y_train == 7).astype('float32')
    y_test = (y_test == 7).astype('float32')
    
    train_idx = np.random.choice(x_train.shape[0], train_size, replace=False)
    test_idx = np.random.choice(x_test.shape[0], test_size, replace=False)

    X = x_train[train_idx].reshape(-1, img_rows * img_cols)
    Y = y_train[train_idx]
    test_X = x_test[test_idx].reshape(-1, img_rows * img_cols)
    test_Y = y_test[test_idx]

    return X, Y, test_X, test_Y

def evaluate_mnist(X, Y, test_X, test_Y, pca_size = 30):
    vals = [evaluate_mnist_once(X, Y, test_X, test_Y, pca_size)[0] for i in range(0, 3)]
    return np.average(vals, axis=0)
    
def evaluate_mnist_once(X, Y, test_X, test_Y, pca_size = 30):
    batch_size = 16
    epochs = 200

    model = Sequential()
    model.add(Dense(2, input_dim=pca_size, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    Y_cat = np_utils.to_categorical(Y, 2)
    test_cat = np_utils.to_categorical(test_Y, 2)
    if len(test_X) > 0:
        model.fit(X, Y_cat,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(test_X, test_cat))
    else:
        model.fit(X, Y_cat,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0)
    if len(test_X) > 0:
        score = model.evaluate(test_X, test_cat, verbose=0)
    else:
        score = []
    
    return score, model # loss, accuracy

def flip_mnist_17(X, Y, factor = 0.2, pca_size=30):
    _, model = evaluate_mnist_once(X, Y, [], [], pca_size=pca_size)
    corrects = np.nonzero(np.argmax(model.predict(X), axis=1) == Y)
    idx_to_flip = np.random.choice(corrects[0], size=int(X.shape[0] * factor), replace=False)
    Y_flipped = np.copy(Y)
    Y_flipped[idx_to_flip] = 1 - Y_flipped[idx_to_flip]
    return Y_flipped