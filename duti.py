import numpy as np
import math
from sklearn import linear_model
from scipy.interpolate import Rbf

np.set_printoptions(precision=8, suppress=True, threshold=1000)

def rbf(a, b, sigma):
    n = a.shape[0]
    m = b.shape[0]
    nms = np.sum(a.T * a.T, axis=0)
    mms = np.sum(b.T * b.T, axis=0)
    D = np.outer(-nms.transpose(), np.ones((1, m))) - np.outer(np.ones((n, 1)), mms)
    D += 2 * np.dot(a, b.T)
    return np.exp(D / (2 * sigma * sigma))

def regression(trainX, trainY, trustX, trustY, confidence, lam, sig):
    n = trainX.shape[0]
    m = trustX.shape[0]

    w = np.concatenate((confidence / m, np.ones((n, )) / n))
    #K = Rbf(trainX, trainX, epsilon=sig, function='gaussian')
    #K_tilde = Rbf(trustX, trainX, epsilon=sig, function='gaussian')
    K = rbf(trainX, trainX, sig)
    K_tilde = rbf(trustX, trainX, sig)
    B = np.dot(K, np.linalg.inv(K + n * lam * np.identity(n))) - np.identity(n)
    A = np.dot(K_tilde, np.linalg.inv(K + n * lam * np.identity(n)))
    
    X = np.concatenate((A, B))
    Y = np.concatenate((trustY - np.dot(A, trainY), np.dot(-B, trainY)))

    lasso = linear_model.Lasso(max_iter=5000)
    lasso.fit(X, Y)

    print(sum(abs(i) > 1e-6 for i in lasso.coef_))

X = np.arange(0, math.pi * 2, 0.01)
X.shape = (X.shape[0], 1)
Y = np.sin(X)
Y.shape = (Y.shape[0], 1)

XX = np.arange(0, math.pi * 2, 0.1)
XX.shape = (XX.shape[0], 1)
YY = np.sin(XX)
YY.shape = (YY.shape[0], 1)

regression(X, Y, XX, YY, np.ones((XX.shape[0], )), 0.0, 0.7)
