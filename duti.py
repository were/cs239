import numpy as np
import tensorflow as tf
import math

np.set_printoptions(precision=8, suppress=True, threshold=1000)

def rbf(a, b, sigma):
    #res[i,j] = np.square(np.sum(a[i,:] - b[j,:]))

    n = a.shape[0]
    m = b.shape[0]

    aa = np.sum(a * a, axis=1)
    bb = np.sum(b * b, axis=1)
    res = np.outer(aa, np.ones((m, ))) + np.outer(np.ones((n, )), bb)
    res -= 2 * np.dot(a, b.T)

    return np.exp(-res / 2.0 / sigma / sigma)

def regression(trainX, trainY, trustX, trustY, confidence, lam, sig):
    n = trainX.shape[0]
    m = trustX.shape[0]

    w = np.concatenate(([100.0] * m, np.ones((n, )) / n))
    assert trainX.shape[1] == trustX.shape[1]

    K = rbf(trainX, trainX, sig)
    K_tilde = rbf(trustX, trainX, sig)
    B = np.dot(K, np.linalg.inv(K + n * lam * np.identity(n))) - np.identity(n)
    A = np.dot(K_tilde, np.linalg.inv(K + n * lam * np.identity(n)))

    X = np.concatenate((A, B))
    Y = np.concatenate((trustY - np.dot(A, trainY), np.dot(-B, trainY)))

    lasso = linear_model.Lasso(max_iter=10000)
    lasso.fit(X, Y)

    return lasso.coef_
