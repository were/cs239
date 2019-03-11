# This file is partially from Influence Function repository
# The original files could be find at https://github.com/kohpangwei/influence-release/tree/master/influence

from __future__ import print_function

import numpy as np
import os
import time

import IPython
from scipy.stats import pearsonr

import duti as duti


def get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test):
    def try_check(idx_to_check, label):
        Y_train_fixed = np.copy(Y_train_flipped)
        Y_train_fixed[idx_to_check] = Y_train[idx_to_check]
        model.update_train_x_y(X_train, Y_train_fixed)
        model.train()
        check_num = np.sum(Y_train_fixed != Y_train_flipped)
        check_loss, check_acc = model.sess.run(
            [model.loss_no_reg, model.accuracy_op], 
            feed_dict=model.all_test_feed_dict)

        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' % (
            label, check_num, check_loss, check_acc))
        return check_num, check_loss, check_acc
    return try_check


def test_mislabeled_detection_batch(
    model,
    X_train, Y_train,
    Y_train_flipped,
    X_test, Y_test,
    train_losses, train_loo_influences,
    num_flips, num_checks):

    assert num_checks > 0
    check_per_iter = int(num_checks / 10)

    num_train_examples = Y_train.shape[0] 
    
    try_check = get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test)

    # Bootstrap with loss
    idx_to_boot = np.argsort(np.abs(train_losses))[-int(num_checks / 2):]
    X_trusted = X_train[idx_to_boot]
    Y_trusted = Y_train[idx_to_boot]
    Y_fli_trus = Y_train_flipped[idx_to_boot]
    all_checked = idx_to_boot
    print("DUTI(debug) checked: ", all_checked)
    try_check(all_checked, 'DUTI(debug)')
    
    while X_trusted.shape[0] < num_checks:
        count = check_per_iter
        if count + X_trusted.shape[0] > num_checks:
            count = num_checks - X_trusted.shape[0]
        # Pick by DUTI delta
        delta = duti.regression(X_train, np.array([Y_train_flipped]).T,
                    X_trusted, np.array([Y_trusted]).T,
                    np.ones((X_trusted.shape[0], )) * 300, 3.8e-6, 0.8)
        delta = np.array([
            0 if (d[0] < 0 and v < 0.5) or (d[0] > 0 and v >= 0.5)
            else d[0] for (d, v) in zip(delta, Y_train_flipped)])
        order = np.argsort(np.abs(delta))
        order = order[~np.in1d(order, all_checked)]
        
        #print(Y_train[order[-count:]], Y_train_flipped[order[-count:]], delta[order[-count:]])
        idx_to_check = order[-count:]
        X_trusted = np.concatenate((X_trusted, X_train[idx_to_check]))
        Y_trusted = np.concatenate((Y_trusted, Y_train[idx_to_check]))
        Y_fli_trus = np.concatenate((Y_fli_trus, Y_train_flipped[idx_to_check]))
        all_checked = np.concatenate((all_checked, idx_to_check))
        print("DUTI(debug) checked: ", all_checked)
        try_check(all_checked, 'DUTI(debug)')
    
    fixed_duti_results = try_check(all_checked, 'DUTI')
    return fixed_duti_results