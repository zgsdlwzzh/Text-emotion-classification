import numpy as np
import tensorflow as tf


class ATT_CNN_BiLstm():
    """
    A C-LSTM classifier for text classification
    Reference: A C-LSTM Neural Network for Text Classification
    """
    def __init__(self,sequence_length,vocab_size,embedding_size,filter_sizes,
                 num_filters,hidden_size,num_classes,attention_size,l2_reg_lambda=0.0):

        # Placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None,num_classes], name='input_y')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # L2 loss
        l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)       # [None,sequence_length,embedding_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)   # [None,sequence_length,embedding_size,1]

        conv_outputs = []
        max_feature_length = sequence_length - max(filter_sizes) + 1
        # Convolutional layer with different lengths of filters in parallel
        # No max-pooling
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope('conv-%s' % filter_size):
                # [filter size, embedding size, channels, number of filters]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # Convolution
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')   # [None,sequence_length-filter_shape+1,1,num_filters]

                # Remove channel dimension
                h_reshape = tf.squeeze(h, [2])                         # [None,sequence_length-filter_shape+1,num_filters]
                # Cut the feature sequence at the end based on the maximum filter length
                h_reshape = h_reshape[:, :max_feature_length, :]       # [None,sequence_length-max(filter_shape)+1,num_filters]

                conv_outputs.append(h_reshape)

        # Concatenate the outputs from different filters
        if len(filter_sizes) > 1:
            rnn_inputs = tf.concat(conv_outputs, -1)     # # [None,sequence_length-filter_shape+1,num_filters*3]
        else:
            rnn_inputs = h_reshape

        # Bi-LSTM
        with tf.name_scope("Bi-LSTM"):
            # forword lstm
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,state_is_tuple=True,name='lstm_fw_cell'),
                output_keep_prob=self.rnn_dropout_keep_prob)


            # backword lstm
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,state_is_tuple=True,name='lstm_bw_cell'),
                output_keep_prob=self.rnn_dropout_keep_prob)


            # outputs 是一个元组(output_fw,output_bw)  两个元素维度都是[None,sequence_length,hidden_size]
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                     rnn_inputs,
                                                                     dtype=tf.float32
                                                                     )

            output = tf.concat(outputs, 2)         # [None,sequence_length,2*hidden_size]

        # Attention layer
        with tf.name_scope("Attention_layer"):
            self.attn_output,self.alphas=self.Attention(output,attention_size)

        # Dropout
        with tf.variable_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.attn_output, self.dropout_keep_prob)

        # Fully connected layer
        with tf.name_scope('output'):
            W = tf.Variable(tf.truncated_normal(shape=[hidden_size * 2, num_classes],stddev=0.1),name='out_W')  # Hidden size is multiplied by 2 for Bi-RNN
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]),name='out_b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def Attention(self,inputs, attention_size):
        hidden_size = inputs.shape[2].value
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)    # [None,sequence_length,attention_size]

        vu = tf.tensordot(v, u_omega, axes=1, name='vu')     # [None,sequence_length]

        alphas = tf.nn.softmax(vu, name='alphas')            # [None,sequence_length]

        # the result has (B,D) shape
        output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)   # [None,2*hidden_size]

        return output, alphas

# import sys
# sys.path.append(r'D:\Python\Python37\study\论文')
# from utils.params_utils import *
# from model.CNN_LSTM import *
# params=get_params()
# model = ATT_CNN_BiLstm(sequence_length=params['sequence_length'],
#                     vocab_size=params['vocab_size'],
#                     embedding_size=params['embedding_size'],
#                     filter_sizes=list(map(int, params['filter_sizes'].split(","))),
#                     num_filters=params['num_filters'],
#                     hidden_size=params['hidden_size'],
#                     num_classes=params['num_classes'],
#                     attention_size=params['attention_size'],
#                     l2_reg_lambda=params['l2_reg_lambda'])