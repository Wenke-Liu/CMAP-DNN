import sys
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime

import fc
import data_input

"""
Global Variables
"""
TST_FILE = None
ENS_DIR = None
PROB_FILE = None
OUTPUT = False
PREFIX = 'test'


def make_ensemble(model_paths):
    """
    make an ensemble of models
    """
    ensemble = []
    for model_path in model_paths:
        model = fc.FC(meta_graph=model_path, save_graph_def=False)
        tf.reset_default_graph()
        print('Including model: {}'.format(model_path), flush=True)
        ensemble.append(model)
    print('Ensemble of {} constructed.'.format(len(ensemble)))
    return ensemble


def get_prob(models, test_data, batch_size=64):
    """
    Get probabilities with inference
    """
    assert test_data.num_examples > batch_size, "Sample size is smaller than batch size!"
    n_batch = math.floor(test_data.num_examples / batch_size)
    print('Number of samples tested: {}'.format(n_batch * batch_size))
    labels = []
    probs = []
    for i in range(n_batch):
        x, y = test_data.next_batch(batch_size)
        labels.append(y)
        prob_list = []
        for model in models:
            prob_list.append(model.inference(x))
        prob = np.mean(prob_list, axis=0)
        probs.append(prob)
    all_labels = np.concatenate(labels)
    all_probs = np.concatenate(probs)
    return all_labels, all_probs


def pred_rank(probs, y, n=5):
    """
    get the top-n performance from probabilities
    """
    lab_prob = probs[(range(y.shape[0]), y)]  # probability of the true class
    prob_order = np.argsort(lab_prob, axis=1)  # order of the probabilities
    prob_rank = np.argsort(prob_order, axis=1)  # rank of probs
    lab_rank = prob_rank[(range(y.shape[0]), y)]  # rank of the true class

    return lab_rank


def main():
    if TST_FILE and ENS_DIR:  # inference from test data and saved model
        print('Loading test data: {}'. format(TST_FILE), flush=True)
        test = data_input.load_test(data_file=TST_FILE, labels=True, skip_rows=1)
        print('Test sample size: {}'.format(test.num_examples))

        model_paths = []
        for i in range(5):
            subdir = ENS_DIR + '/model' + str(i) + '/out/'
            model_paths.extend([fn for fn in os.listdir(subdir) if 'meta' in fn])
            model_paths[i] = subdir + model_paths[i].replace('.meta', '')

        print('Model paths:')
        print("\n".join(model_paths))
        fc_ensemble = make_ensemble(model_paths)
        y_true, y_probs = get_prob(fc_ensemble, test)

    elif TST_FILE and PROB_FILE:
        _, y_true = data_input.numpy_input(data_file=TST_FILE)
        y_probs = data_input.iter_loadtxt(PROB_FILE, delimiter='\t')

    else:
        sys.exit('Test file required.')

    rank = pred_rank(y_probs, y_true, n=1)
    rank1 = (rank < 1).astype(int)
    rank5 = (rank < 5).astype(int)

    y_onehot = np.zeros((y_true.size, y_true.max() + 1))
    y_onehot[np.arange(y_true.size), y_true] = 1

    classes = np.unique(y_true)
    y_onehot = y_onehot[:, classes]
    y_scores = y_probs[:, classes]

    print('Rank 1 test accuracy: {}'.format(np.sum(rank1) / rank1.shape[0]), flush=True)
    print('Rank 5 test accuracy: {}'.format(np.sum(rank5) / rank5.shape[0]), flush=True)

    pr = average_precision_score(y_onehot, y_scores)
    print('Average PR (macro): {}'.format(np.mean(pr)))
    print('Average PR (micro): {}'.format(average_precision_score(y_onehot, y_scores, average='micro')))

    roc = roc_auc_score(y_onehot, y_scores)
    print('Average AU-ROC (macro): {}'.format(np.mean(roc)))
    print('Average AU-ROC (macro): {}'.format(roc_auc_score(y_onehot, y_scores, average='micro')))

    if OUTPUT:
        output = np.hstack((classes, rank, pr, roc))
        np.savetxt(fname=datetime.now().isoformat()[0:10] + '_' + PREFIX + '_metrics.txt',
                   X=output,
                   delimiter='\t',
                   comments='',
                   header='y_true' + '\t' + 'rank' + '\t' + 'pr' + '\t' + 'roc')


if __name__ == "__main__":
    main()
