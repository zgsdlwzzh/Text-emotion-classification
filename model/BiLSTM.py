import tensorflow as tf

class BiLSTM:
    def __init__(
            self,sequence_length,num_classes,vocab_size,
            embedding_size,hidden_size,batch_size,l2_reg_lambda=0.0):

        # Placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)    # [None,sequence_length,embedding_size]

        # Bi-LSTM
        with tf.name_scope("Bi-LSTM"):
            # forword lstm
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,state_is_tuple=True,name='lstm_fw_cell'),
                output_keep_prob=self.dropout_keep_prob)


            # backword lstm
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size,state_is_tuple=True,name='lstm_bw_cell'),
                output_keep_prob=self.dropout_keep_prob)

            # self._initial_state_fw=lstm_fw_cell.zero_state(batch_size,dtype=tf.float32)
            # self._initial_state_bw = lstm_bw_cell.zero_state(batch_size, dtype=tf.float32)

            # outputs 是一个元组(output_fw,output_bw)  两个元素维度都是[None,sequence_length,hidden_size]
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                     self.embedded_chars,
                                                                     dtype=tf.float32
                                                                     )

            output = tf.concat(outputs, 2)         # [None,sequence_length,2*hidden_size]

            # 取出最后时间步的输出作为全连接的输入  [None,2*hidden_size]
            final_output = output[:, -1, :]
            output_size = hidden_size * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
            output = tf.reshape(final_output, [-1, output_size])  # reshape成全连接层的输入维度 [None,2*hidden_size]


            # Softmax output layer
            with tf.name_scope('softmax'):
                softmax_w = tf.Variable(tf.truncated_normal(shape=[output_size, num_classes],stddev=0.1),name='softmax_w')
                softmax_b = tf.Variable(tf.truncated_normal(shape=[num_classes]),name='b')

                # L2 regularization for output layer
                l2_loss += tf.nn.l2_loss(softmax_w)
                l2_loss += tf.nn.l2_loss(softmax_b)

                # logits  [None,num_classes]
                self.logits = tf.nn.xw_plus_b(output, softmax_w,softmax_b,name='logits')
                predictions = tf.nn.softmax(self.logits)
                self.predictions = tf.argmax(predictions, 1, name='predictions')


            # Loss
            with tf.name_scope('loss'):
                losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope('accuracy'):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name='accuracy')



# bilstm=BiLSTM(sequence_length=params['sequence_length'],
#             num_classes=params['num_classes'],
#             vocab_size=params['vocab_size'],
#             embedding_size=params['embedding_size'],
#             hidden_size=params['hidden_size'],
#             batch_size=params['batch_size'],
#             l2_reg_lambda=params['l2_reg_lambda'])

# reuse=tf.get_variable_scope().reuse
# initial_state_fw=self._initial_state_fw,
# initial_state_bw=self._initial_state_bw