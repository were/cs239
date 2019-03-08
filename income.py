import duti, pandas, subprocess, os, random
import numpy as np
subprocess.check_output(['unzip', '-o', 'adult.csv.zip'])

a = pandas.read_csv('./adult.csv')

discret = {}
norms = {}

for i in a:
    if len(set(a[i])) < 50:
        discret[i] = list(set(a[i]))
        norms[i] = len(discret[i])
    else:
        norms[i] = max(set(a[i]))

def obj_to_np(universe, sample):
    X = []
    Y = []
    for i in sample:
        row = []
        num, obj = i
        for j in universe:
            if j in discret.keys():
                row.append(discret[j].index(obj[j]))
            else:
                row.append(obj[j])
            row[-1] /= norms[j]
        print(row)
        X.append(row[:-1])
        Y.append([0.0, 1.0][int(row[-1])])
    X, Y = np.array(X), np.array(Y)
    Y.shape = (Y.shape[0], 1)
    return X, Y


def random_flip(n, Y):
    noise = []
    for i in range(n):
        idx = random.randint(0, Y.shape[0] - 1)
        while idx in noise:
            idx = random.randint(0, Y.shape[0] - 1)
        Y[idx][0] = 1 - Y[idx][0]
        noise.append(idx)
    return noise

def find_errs(delta, noise, thr=1e-5):
    total = correct = 0
    found = []
    for i, j in enumerate(delta):
        if abs(j) >= thr:
            total += 1
            if i in noise:
                correct += 1
                found.append(i)
    if total:
        print('Recall: %d / %d = %.2f' % (correct, len(noise), float(correct) / len(noise)))
        print('Precision: %d / %d = %.2f' % (correct, total, float(correct) / total))
    else:
        print('No error found!')

def experiment1():
    list_a = list(a.iterrows())
    train_a = random.sample(list_a, 1000)
    trust_a = random.sample(list_a, 100)

    X, Y = obj_to_np(a, train_a)
    noise = random_flip(100, Y)
    XX, YY = obj_to_np(a, trust_a)

    deltas = duti.regression(X, Y, XX, YY, np.ones((100, )) * 200., 1e-6, 0.7)

    find_errs(deltas, noise, 1e-4)


experiment1()

os.remove('./adult.csv')
