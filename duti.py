import numpy as np
import tensorflow as tf
import math
from sklearn import linear_model

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

def lasso_with_weight(X, Y, weight, alpha=1.0, max_iter=1000):
    X = X.astype('float32')
    Y = Y.astype('float32')
    weight = weight.astype('float32')

    tf_x = tf.placeholder(dtype=tf.float32, shape=X.shape, name='X')
    tf_y = tf.placeholder(dtype=tf.float32, shape=Y.shape, name='Y')
    tf_w = tf.placeholder(dtype=tf.float32, shape=weight.shape, name='W')
    lr = tf.placeholder(dtype=tf.float32, shape=[], name='LR')

    tf.set_random_seed(114514)

    A = tf.Variable(tf.random_normal(shape=[X.shape[1], Y.shape[1]]), name='ModelParam')
    #b = tf.Variable(tf.random_normal(shape=(Y.shape[0], )))

    tf_predict = tf.matmul(tf_x, A)
    #tf_predict = tf.add(tf_predict, b)

    diff = tf.square(tf.subtract(tf_y, tf_predict))
    w_diff = tf.multiply(diff, tf_w)
    #w_diff = diff
    mis_predict = tf.reduce_mean(tf.square(w_diff))
    vec_length  = tf.multiply(tf.norm(A, ord=1), alpha)
    lasso_loss = tf.add(mis_predict, vec_length)

    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_step = opt.minimize(lasso_loss)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        last_loss = 114514
        learning_rate = 0.001
        mis, vec = sess.run([mis_predict, vec_length], feed_dict={tf_x: X, tf_y:Y, tf_w: weight})
        for i in range(10000):
            train = sess.run(train_step, feed_dict={tf_x: X, tf_y:Y, tf_w: weight, lr: learning_rate})
            current_loss = sess.run(lasso_loss, feed_dict={tf_x: X, tf_y:Y, tf_w: weight})

            if (i + 1) % 500 == 0:
                print(i + 1, ':', current_loss, learning_rate)
                mis, vec = sess.run([mis_predict, vec_length], feed_dict={tf_x: X, tf_y:Y, tf_w: weight})
                print('Misprediction: ', mis, 'Delta-sum: ', vec)
                if last_loss - current_loss < learning_rate:
                    learning_rate *= 0.1
                    if learning_rate < 1e-6:
                        break
                last_loss = current_loss

        return sess.run(A)



def regression(trainX, trainY, trustX, trustY, confidence, lam, sig, alpha=1.0):
    n = trainX.shape[0]
    m = trustX.shape[0]

    w = np.concatenate((confidence, np.ones((n, ))))
    if trustY.shape[1] != 1:
        w = np.repeat(w, trainY.shape[1], axis=1)
    assert trainX.shape[1] == trustX.shape[1]

    K = rbf(trainX, trainX, sig)
    K_tilde = rbf(trustX, trainX, sig)
    B = np.dot(K, np.linalg.inv(K + n * lam * np.identity(n))) - np.identity(n)
    A = np.dot(K_tilde, np.linalg.inv(K + n * lam * np.identity(n)))

    X = np.concatenate((A, B))
    Y = np.concatenate((trustY - np.dot(A, trainY), np.dot(-B, trainY)))

    #lasso = linear_model.Lasso(max_iter=10000)
    #lasso.fit(X, Y)
    #return lasso.coef_

    return lasso_with_weight(X, Y, w, alpha=alpha)
