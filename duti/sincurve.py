import duti, pandas, subprocess, os, math, random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(114514)

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
    Y[l:r] = -Y[l:r]
    return list(range(l, r))

def find_errs(delta, noise, fname):
    total = correct = 0
    indexed = [(j, i) for i, j in enumerate(delta)]
    indexed.sort(reverse=True, key=lambda a: abs(a[0]))

    n = len(delta)
    nn = len(noise)
    k = 0
    x, y = [], []
    while k < n and correct < nn:
        j, i = indexed[k]
        if i in noise:
            correct += 1
        x.append(float(correct) / len(noise))
        y.append(float(correct) / (k + 1))
        k += 1
    
    with open(fname, 'w') as f:
        f.write(str(x) + '\n')
        f.write(str(y) + '\n')

    return indexed


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

def extend_trust(indexed, X, Y, XX, YY, noise, n, confirm):
    for i in range(min(n, len(indexed))):
        _, idx = indexed[i]
        if idx in noise:
            Y[idx, 0] = np.sin(X[idx, 0])
            if confirm:
                XX = np.append(XX, X[idx])
                YY = np.append(YY, Y[idx])
        else:
            XX = np.append(XX, X[idx])
            YY = np.append(YY, Y[idx])

    XX.shape = (XX.shape[0], 1)
    YY.shape = (YY.shape[0], 1)

    return XX, YY

def expreiment1(w):
    X, Y, XX, YY = init_data(500, ratio=0.1)
    noise = uniform_noise(50, Y)
    delta = duti.regression(X, Y, XX, YY, np.ones((XX.shape[0], )) * w, 3.8e-6, 0.7)
    find_errs(delta, noise, 'uniform%d' % w)

def expreiment2(w):
    subprocess.check_output(['mkdir', '-p', 'ranged'])
    X, Y, XX, YY = init_data(500, ratio=0.05)
    noise = range_noise(50, 100, Y)

    delta = duti.regression(X, Y, XX, YY, np.ones((XX.shape[0], )) * w, 3.8e-6, 0.6)
    indexed = find_errs(delta, noise, 'ranged/ranged-origin')

    XX0, YY0 = extend_trust(indexed, X, Y, XX, YY, noise, 50, True)
    delta = duti.regression(X, Y, XX0, YY0, np.ones((XX.shape[0], )) * w, 3.8e-6, 0.6)
    indexed = find_errs(delta, noise, 'ranged/ranged-correct')

    XX1, YY1 = extend_trust(indexed, X, Y, XX, YY, noise, 50, True)
    delta = duti.regression(X, Y, XX1, YY1, np.ones((XX.shape[0], )) * w, 3.8e-6, 0.6)
    indexed = find_errs(delta, noise, 'ranged/ranged-confirm')

#expreiment1(1)
#expreiment1(25)
#expreiment1(50)
#expreiment1(100)
#expreiment1(200)
#expreiment1(300)
#expreiment2(50)
expreiment2(100)
#expreiment2(300)
