from datetime import datetime
import os
import re
import sys
import numpy as np
import tensorflow as tf

import layers


class FC:
    """
    DNN with fully connected layers
    """
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 save_graph_def=True, log_dir="./log"):
        """(Re)build a symmetric fully connected model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a FC with 1000-D inputs, 10-D latents

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.architecture = architecture
        self.__dict__.update(FC.DEFAULTS, **d_hyperparams)
        
        self.sesh = tf.Session()

        if not meta_graph:  # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(FC.RESTORE_KEY, handle)
            self.sesh.run(tf.global_variables_initializer())

        else: # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_fc_")
            self.datetime = "{}_reloaded".format(model_datetime)
            arch_par, hyper_par = model_name.split("_lr_")
            *model_architecture, _ = re.split("_|-", arch_par)
            self.architecture = [int(n) for n in model_architecture]

            # rebuild graph
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(FC.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.y_in, 
         self.dropout_,
         self.h_encoded, self.pred,
         self.cost, self.global_step, self.train_op, self.merged_summary) = handles

        if save_graph_def: # tensorboard
            try:
                os.mkdir(log_dir + '/training')
                os.mkdir(log_dir + '/validation')
            except(FileExistsError):
                pass
            self.train_logger = tf.summary.FileWriter(log_dir + '/training', self.sesh.graph)
            self.valid_logger = tf.summary.FileWriter(log_dir + '/validation', self.sesh.graph)
    
    def __del__(self):
        print('FC object destructed.')
        
    
    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")
    
        y_in = tf.placeholder(dtype=tf.int16,name="y")
        
        onehot_labels = tf.one_hot(indices=tf.cast(y_in, tf.int32), depth=4314)
        
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        h_encoded = x_in
        for idx, hidden_size in enumerate(self.architecture[1:]):
            h_encoded = layers.fc_dropout(x=h_encoded,
                                          size=int(hidden_size),
                                          scope="h"+str(idx)+"fc_drop_"+ str(hidden_size),
                                          dropout=dropout)

        logits = layers.fc_dropout(h_encoded, scope="classifcation", 
                                      size=4314,
                                      activation=tf.identity,
                                      dropout=dropout)
        
        pred = tf.nn.softmax(logits, name="prediction")

        # classification loss: cross-entropy with the gene knockdown labels
        pred_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
        
        tf.summary.scalar("pred_cost",pred_loss)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            # average over minibatch                        
            cost = pred_loss
            cost += l2_reg

            tf.summary.scalar("cost",cost)

        # optimization
        global_step = tf.Variable(0, trainable=False)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):  
                 
            train_op = tf.contrib.layers.optimize_loss(
                       loss=cost,
                       learning_rate=self.learning_rate,
                       global_step=global_step,
                       optimizer="Adam")

        merged_summary = tf.summary.merge_all()
        
        return (x_in, y_in, 
                dropout, 
                h_encoded, pred,
                cost, global_step, train_op, merged_summary)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    def inference(self, x):
        """predict class from input"""
        feed_dict = {self.x_in:x}
        return self.sesh.run(self.pred, feed_dict=feed_dict)
    
    def bottleneck(self, x):
        feed_dict = {self.x_in: x}
        return self.sesh.run('h2fc_drop_500/fully_connected/BiasAdd:0',feed_dict=feed_dict)

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=True, save_log=True,
              outdir="./out"):
        
        if save:
            saver = tf.train.Saver(tf.global_variables(),max_to_keep=None)

        try:
            err_train=0
            now = datetime.now().isoformat()
            print("------- Training begin: {} -------".format(now),flush=True)

            while True:
                x, y = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.y_in: y,
                             self.dropout_: self.dropout}
                fetches = [self.cost, self.global_step, self.merged_summary, self.train_op]
                cost, i, summary, _ = self.sesh.run(fetches, feed_dict)
                
                if save_log:
                    self.train_logger.add_summary(summary, i)

                err_train += cost

                if i % 10000 == 0 and verbose:
                    print("round {} --> trn cost: ".format(i), cost, flush=True)

                if i % 10000 == 0 and verbose:  # and i >= 10000:
                    
                    if cross_validate:
                        x, y = X.validation.next_batch(128)
                        feed_dict = {self.x_in: x, self.y_in: y}
                        fetches = [self.cost, self.merged_summary]
                        valid_cost,  valid_summary = self.sesh.run(fetches, feed_dict)
                        
                        if save_log:
                            self.valid_logger.add_summary(valid_summary,i)
                        
                        

                        print("round {} --> CV cost: ".format(i), valid_cost, flush=True)
                        
                """
                if i%200000 == 0 and save:
                    interfile=os.path.join(os.path.abspath(outdir), "{}_fc_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                    saver.save(self.sesh, interfile, global_step=self.step)
                """    
                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i),flush=True)
                    
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------".format(now),flush=True)
                    

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_fc_{}_lr_{}_l2_{}".format(
                            self.datetime, "_".join(map(str, self.architecture)),
                            str(self.learning_rate),str(self.lambda_l2_reg)))
                        
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.train_logger.flush()
                        self.train_logger.close()
                        self.valid_logger.flush()
                        self.valid_logger.close()
                    except(AttributeError): # not logging
                        print('Not logging',flush=True)
                        pass
                    
                    break

        except KeyboardInterrupt:
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i), flush=True)
            
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now),flush=True)
            
            if save:
                outfile = os.path.join(os.path.abspath(outdir), "{}_fc_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                saver.save(self.sesh, outfile, global_step=self.step)
            try:
                self.train_logger.flush()
                self.train_logger.close()
                self.valid_logger.flush()
                self.valid_logger.close()
                
            except AttributeError:  # not logging
                print('Not logging',flush=True)

            sys.exit(0)
