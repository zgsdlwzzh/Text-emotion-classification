import tensorflow as tf

class RCNN:
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,
                 hidden_size,output_size,l2_reg_lambda=0.0):

        # Placeholders
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)  # [None,sequence_length,embedding_size]
            embedded_words_ = self.embedded_chars

        # Bi-LSTM
        with tf.name_scope("Bi-LSTM"):
            # forword lstm
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True, name='lstm_fw_cell'),
                output_keep_prob=self.dropout_keep_prob)

            # backword lstm
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True, name='lstm_bw_cell'),
                output_keep_prob=self.dropout_keep_prob)

            # outputs 是一个元组(output_fw,output_bw)  两个元素维度都是[None,sequence_length,hidden_size]
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                     self.embedded_chars,
                                                                     dtype=tf.float32
                                                                     )

            # 对outputs中的fw和bw的结果拼接 [None, sequence_length, hidden_size * 2]
            embedded_words = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出  shape均为[None,sequence_length,hidden_size]
        fw_output, bw_output = tf.split(embedded_words, 2, -1)

        with tf.name_scope("context"):
            shape = [tf.shape(fw_output)[0], 1, tf.shape(fw_output)[2]]     # [None,1,hidden_size]
            context_left = tf.concat([tf.zeros(shape), fw_output[:, :-1]], axis=1, name="context_left")   # [None,sequence_length,hidden_size]
            context_right = tf.concat([bw_output[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("wordRepresentation"):
            word_representation = tf.concat([context_left, embedded_words_, context_right], axis=2)
            word_size = hidden_size * 2 + embedding_size

        with tf.name_scope("text_representation"):
            output_size = output_size
            text_w = tf.Variable(tf.random_uniform([word_size, output_size], -1.0, 1.0), name="text_w")
            text_b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="text_b")

            # tf.einsum可以指定维度的消除运算   [None,sequence_length,output_size]
            text_representation = tf.tanh(tf.einsum('aij,jk->aik', word_representation, text_w) + text_b)

        # 做max-pool的操作，将时间步的维度消失  [None,output_size]
        output = tf.reduce_max(text_representation, axis=1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[output_size, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.truncated_normal(shape=[num_classes]), name='b')

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


# model = RCNN(sequence_length=params['sequence_length'],
#                     num_classes=params['num_classes'],
#                     vocab_size=params['vocab_size'],
#                     embedding_size=params['embedding_size'],
#                     hidden_size=params['hidden_size'],
#                     output_size=params['output_size'],
#                     l2_reg_lambda=params['l2_reg_lambda'])