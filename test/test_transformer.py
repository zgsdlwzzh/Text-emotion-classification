import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import tensorflow as tf
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import time
import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from model.TextCNN import *
from utils.batcher import *
from utils.config import *
from utils.content_to_id import ContentIDlst
from utils.params_utils import *
from train.train import *

def predict(checkpoint_dir_path):
    # graph = tf.Graph()
    graph = tf.get_default_graph()

    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_dir_path))
            start_time=time.time()
            saver.restore(sess, checkpoint_dir_path)
            # # 获取最新的checkpoint
            # checkpoint_file=tf.train.latest_checkpoint(checkpoint_dir_path)
            # saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            # saver.restore(sess,checkpoint_file)

            print('load model use time:',time.time()-start_time)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            emb_dropout_keep_prob = graph.get_operation_by_name("emb_dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                if len(x_test_batch)==params['batch_size']:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, emb_dropout_keep_prob: 1.0,dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

            all_predictions=all_predictions.astype('int64')
            y_true=np.argmax(y_test, axis=1)[:len(all_predictions)]
            # 混淆矩阵
            result_predict=pd.DataFrame(confusion_matrix(y_true,all_predictions))

            return y_true,all_predictions,result_predict


def eval_model(y_true,y_pred):
    p,r,f1,s=precision_recall_fscore_support(y_true,y_pred)
    # 计算总体的平均prediction,recall,f1,support
    tot_p=np.average(p,weights=s)
    tot_r=np.average(r,weights=s)
    tot_f1=np.average(f1,weights=s)
    tot_s=np.sum(s)
    res1=pd.DataFrame({
        u'Precision':p,
        u'Recall':r,
        u'F1':f1,
        u'Support':s
    })
    res2=pd.DataFrame({
        u'Precision':[tot_p],
        u'Recall':[tot_r],
        u'F1':[tot_f1],
        u'Support':[tot_s]
    })
    res2.index=['总体']
    res=pd.concat([res1,res2])
    return res[['Precision','Recall','F1','Support']]

def main(checkpoint_dir_path):
    for step in range(17000, 20001, 500):
        y_true, all_predictions, result_predict = predict(
            os.path.join(checkpoint_dir_path, 'model-{}').format(step))
        print('step {} result_predict:\n'.format(step), result_predict)
        result_eval = eval_model(y_true, all_predictions)
        print('step {} result_eval:\n'.format(step), result_eval)

if __name__=='__main__':
    params = get_params()
    x_train, y_train, x_test, y_test = preprocess(params)
    if params['model']=='transformer':
        checkpoint_dir_path = transformer_checkpoint_dir_path
    elif params['model']=='MultiATT_CNN':
        checkpoint_dir_path = multiatt_cnn_checkpoint_dir_path

    main(checkpoint_dir_path)