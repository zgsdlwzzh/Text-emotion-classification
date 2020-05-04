import sys
sys.path.append(r'D:\Python\Python37\study\论文')
import tensorflow as tf


class ATT_BiLSTM:
    def __init__(
            self,sequence_length,num_classes,vocab_size,embedding_size,
            hidden_size,attention_size,l2_reg_lambda):

        # Placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.emb_dropout_keep_prob = tf.placeholder(tf.float32, name='emb_dropout_keep_prob')
        self.rnn_dropout_keep_prob = tf.placeholder(tf.float32, name='rnn_dropout_keep_prob')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="emb_W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None,sequence_length,embedding_size]

        # Dropout for Word Embedding
        with tf.name_scope('dropout-embeddings'):
            self.embedded_chars = tf.nn.dropout(self.embedded_chars, self.emb_dropout_keep_prob)

        # Bi-LSTM
        with tf.name_scope("Bi-LSTM"):
            # forword lstm
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True, name='lstm_fw_cell'),
                output_keep_prob=self.rnn_dropout_keep_prob)

            # backword lstm
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True, name='lstm_bw_cell'),
                output_keep_prob=self.rnn_dropout_keep_prob)

            # outputs 是一个元组(output_fw,output_bw)  两个元素维度都是[None,sequence_length,hidden_size]
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                                     lstm_bw_cell,
                                                                     self.embedded_chars,
                                                                     dtype=tf.float32
                                                                     )

            output = tf.concat(outputs, 2)  # [None,sequence_length,2*hidden_size]

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



    # att_bilstm= ATT_BiLSTM(sequence_length=params['sequence_length'],
    #                     num_classes=params['num_classes'],
    #                     vocab_size=params['vocab_size'],
    #                     embedding_size=params['embedding_size'],
    #                     hidden_size=params['hidden_size'],
    #                     attention_size=params['attention_size'],
    #                     l2_reg_lambda=params['l2_reg_lambda'])