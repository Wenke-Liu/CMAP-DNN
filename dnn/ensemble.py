#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:55:03 2019

@author: lwk
"""

import os
import numpy as np
import tensorflow as tf

import data_input
import fc


"""
Global variables:
    
"""
TRN_FILE = '/home/fenyopc003/projects/wenke/cmap/data/final/sh_split1_data_trn_500.txt'
VAL_FILE = '/home/fenyopc003/projects/wenke/cmap/data/final/sh_split1_data_val_500.txt'

print("Training data file: " + TRN_FILE, flush=True)
print("Validation data file: " + VAL_FILE, flush=True)

ENS_N = 5

ARCHITECTURE = [978,  # 9669 Bing Genes, 978 landmark genes
                5000, 2000, 500, 1000,  # intermediate encoding
                2000]   # last layer before output

HYPERPARAMS = {
    "batch_size": 128,
    "learning_rate": 1E-4,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-4,
    "nonlinearity": tf.nn.elu,
    "squashing": tf.nn.sigmoid
}

MAX_ITER = np.inf
MAX_EPOCHS = 100
RES_DIR = '/home/fenyopc003/projects/wenke/cmap/1905/res/190820'


def main():
    data = data_input.load_data(TRN_FILE, VAL_FILE, skip_rows=1)
    for i in range(ENS_N):
        MODEL_DIR = RES_DIR + '/model' + str(i)
        METAGRAPH_DIR = MODEL_DIR + '/out'
        LOG_DIR = MODEL_DIR + '/log'
        for DIR in (MODEL_DIR, LOG_DIR, METAGRAPH_DIR):
            try:
                os.mkdir(DIR)
            except FileExistsError:
                pass
        print('Architecture: {}'.format(ARCHITECTURE), flush=True)
        m = fc.FC(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR,save_graph_def=True)
        print('Resetting training epochs', flush=True)
        data.train.reset_epochs()
        print('Resetting validation epochs', flush=True)
        data.validation.reset_epochs()
        m.train(data, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=True,
                verbose=True, save=True, save_log=True,outdir=METAGRAPH_DIR)
        print("Trained!",flush=True)
        del m
        tf.reset_default_graph()
    
    
if __name__ == "__main__":
    tf.reset_default_graph()

    try:
        os.mkdir(RES_DIR)
    except FileExistsError:
        pass
    main()
