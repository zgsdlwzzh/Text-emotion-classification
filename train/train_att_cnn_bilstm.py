import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import tensorflow as tf
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import time
import datetime
from utils.batcher import *
from utils.config import *
from utils.content_to_id import ContentIDlst
from model.ATT_CNN_BiLSTM import *
from utils.params_utils import *

def preprocess(params):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    conid = ContentIDlst(params)
    conid.main()
    x_train=conid.train_reviews_idlst
    y_train=conid.train_labels_onehot
    x_test=conid.test_reviews_idlst
    y_test=conid.test_labels_onehot
    return x_train,y_train,x_test,y_test

def train(x_train, y_train, x_test, y_test,params,
          train_summary_dir_path,test_summary_dir_path,
          checkpoint_dir_path,checkpoint_prefix_path):
    # Training
    # ==================================================
    res={'step':[],'loss':[],'acc':[]}
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            # Define Training procedure
            model = ATT_CNN_BiLstm(sequence_length=params['sequence_length'],
                             vocab_size=params['vocab_size'],
                             embedding_size=params['embedding_size'],
                             filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                             num_filters=params['num_filters'],
                             hidden_size=params['hidden_size'],
                             num_classes=params['num_classes'],
                             attention_size=params['attention_size'],
                             l2_reg_lambda=params['l2_reg_lambda'])
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(model.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)   # 每迭代一个batch,global_step+1

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            out_dir = out_dir_path
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = train_summary_dir_path
            if not os.path.exists(train_summary_dir):
                os.makedirs(train_summary_dir)
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([loss_summary, acc_summary])
            test_summary_dir = test_summary_dir_path
            if not os.path.exists(test_summary_dir):
                os.makedirs(test_summary_dir)
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = checkpoint_dir_path
            checkpoint_prefix = checkpoint_prefix_path
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=params['num_checkpoints'])

            # Write vocabulary
            # vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.rnn_dropout_keep_prob:params['rnn_dropout_keep_prob'],
                  model.dropout_keep_prob: params['dropout_keep_prob']
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                res['step'].append(step)
                res['loss'].append(loss)
                res['acc'].append(accuracy)
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def test_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  model.input_x: x_batch,
                  model.input_y: y_batch,
                  model.rnn_dropout_keep_prob: 1.0,
                  model.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, test_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), params['batch_size'], params['num_epochs'])
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                # if current_step % params['evaluate_every'] == 0:
                #     print("\nEvaluation:")
                #     test_step(x_test, y_test, writer=test_summary_writer)
                #     print("")
                if current_step % params['checkpoint_every'] == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step==10000:
                    return


def main(save_loss_acc_path):
    params = get_params()
    x_train, y_train,  x_test, y_test = preprocess(params)

    start_time = time.time()
    train(x_train, y_train, x_test, y_test,params,
          att_cnn_bilstm_train_summary_dir_path,att_cnn_bilstm_test_summary_dir_path,
          att_cnn_bilstm_checkpoint_dir_path,att_cnn_bilstm_checkpoint_prefix_path)
    print('train model ues time:',time.time()-start_time)

    df = pd.DataFrame(res)
    df.to_csv(os.path.join(loss_acc_path, save_loss_acc_path), index=False)

if __name__ == '__main__':
    res = {'step': [], 'loss': [], 'acc': []}
    main('att_cnn_bilstm')