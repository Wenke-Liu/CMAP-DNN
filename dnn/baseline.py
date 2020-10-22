import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from data_input import iter_loadtxt

"""
Baseline models with softmax regression
"""

"""
Global variables
"""
PREFIX = 'split1'
TRN_FILE = '../data/final/sh_split1_data_trn.txt'
VAL_FILE = '../data/final/sh_split1_data_val.txt'
TST_FILE = '../data/final/sh_split1_data_tst.txt'
ENS_N = 5
MAX_EPOCHS = 3
BATCH_SIZE = 128


def numpy_input(data_file, delimiter='\t', skiprows=0):
    print('Data file: {}'.format(data_file))
    dat = iter_loadtxt(filename=data_file, delimiter=delimiter, skiprows=skiprows)
    x = dat[:, :-1]
    y = dat[:, -1]
    print('X size: {}'.format(x.shape))
    print('y size: {}'.format(y.shape))
    return x, y


def compile_and_fit(model, trn, val, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    def accuracy(y_true, y_pred):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=1)

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.sparse_categorical_crossentropy,
                           accuracy,
                           tf.keras.metrics.sparse_top_k_categorical_accuracy])

    history = model.fit(x=trn[0],
                        y=trn[1],
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        validation_data=val,
                        callbacks=[early_stopping],
                        verbose=2)
    return history


def main():
    print('Loading training data')
    trn_dat = numpy_input(data_file=TRN_FILE, skiprows=1)

    if VAL_FILE:
        print('Loading validation data')
        val_dat = numpy_input(data_file=VAL_FILE)
    else:
        print('Training with no validation data')
        val_dat = None

    print('Loading testing data')
    tst_dat = numpy_input(data_file=TST_FILE)
    ens = {}
    histories = []
    eval_metrics = []
    probs = []
    for i in range(ENS_N):
        ens[i] = tf.keras.Sequential([tf.keras.layers.Dense(units=np.unique(trn_dat[1]).shape[0],
                                                            activation='softmax')])
        history = compile_and_fit(ens[i], trn=trn_dat, val=val_dat)
        histories.append(history)
        ens[i].summary()
        print('Model {} out of {} trained.'.format(i, ENS_N))
        eval_metric = ens[i].evaluate(x=tst_dat[0], y=tst_dat[1], batch_size=4)
        print('Evaluation metrics on model {}: {}'.format(i, eval_metric))
        eval_metrics.append(eval_metric)
        prob = ens[i].predict(x=tst_dat[0], batch_size=4)
        probs.append(prob)
        print('Probabilities: {}'.format(prob.shape))

    np.savetxt(fname=datetime.now().isoformat()[0:10] + '_' + PREFIX + '_' + 'ensemble' + '_probs.txt',
               X=np.mean(probs, axis=0),
               delimiter='\t')

    metric_ave = np.mean(eval_metrics, axis=0)
    print('Mean cross-entropy: {}'.format(metric_ave[0]))
    print('Mean accuracy: {}'.format(metric_ave[1]))
    print('Mean top 5 accuracy: {}'.format(metric_ave[2]))


if __name__ == "__main__":
    main()

