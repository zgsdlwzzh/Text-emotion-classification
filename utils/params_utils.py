import sys
sys.path.append(r'D:\Python\Python37\study\论文')

import tensorflow as tf

def get_params():
    params = {"model":'TextCNN',
              "sequence_length": 230,
              "num_classes":3,
              # "vocab_size":14562,
              "vocab_size":10881,
              "embedding_size":300,
              "filter_sizes":"3,4,5",
              "num_filters":32,   # Number of filters per filter size. For CNN
              "l2_reg_lambda":0.001,
              "num_checkpoints":11,  # Number of checkpoints to store
              "dropout_keep_prob":0.5,
              "batch_size":32,
              "num_epochs":200,
              "evaluate_every":100,  # Evaluate model on dev set after this many steps
              "checkpoint_every":500,  # Save model after this many steps
              "hidden_size":256,
              "attention_size":200,
              "emb_dropout_keep_prob":0.7,
              "rnn_dropout_keep_prob":0.7,
              "num_layers":2,
              "output_size":128,
              "num_blocks": 1,
              "ln_epsilon": 1e-8,
              "num_heads": 8,
              "Wl":0.3
              }
    return params


