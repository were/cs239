import duti, pandas, subprocess, os, math, random
import numpy as np

def uniform_noise(n, Y):
    noise = []
    for i in range(n):
        idx = random.randint(0, Y.shape[0] - 1)
        while idx in noise:
            idx = random.randint(0, Y.shape[0] - 1)
        Y[idx][0] += random.random()
        noise.append(idx)
    return noise

def range_noise(l, r, Y):
    Y[l:r] += np.random.randn(r - l) - 0.5
    return list(range(l, r))

def find_errs(delta, noise):
    total = correct = 0
    for i, j in enumerate(delta):
        if abs(j) > 1e-5:
            total += 1
            correct += i in noise
    if total:
        print('Recall: %d / %d = %.2f' % (correct, len(noise), float(correct) / len(noise)))
        print('Precision: %d / %d = %.2f' % (correct, total, float(correct) / total))
    else:
        print('No error found!')


def init_data(n, ratio=0.05):
    step  = math.pi * 2 / n

    X = np.arange(0, math.pi * 2, step)
    Y = np.sin(X)
    X.shape = (X.shape[0], 1)
    Y.shape = (Y.shape[0], 1)

    XX = np.arange(0, math.pi * 2, step / ratio)
    YY = np.sin(XX)
    XX.shape = (XX.shape[0], 1)
    YY.shape = (YY.shape[0], 1)

    return X, Y, XX, YY

def expreiment1():
    X, Y, XX, YY = init_data(500, ratio=0.1)
    noise = uniform_noise(20, X)
    delta = duti.regression(X, Y, XX, YY, np.ones((XX.shape[0], )), 0.0, 0.8)
    find_errs(delta, noise)

def expreiment2():
    X, Y, XX, YY = init_data(500, ratio=0.1)
    noise = uniform_noise(20, X)
    delta = duti.regression(X, Y, XX, YY, np.ones((XX.shape[0], )), 0.0, 0.8)
    find_errs(delta, noise)

expreiment1()
