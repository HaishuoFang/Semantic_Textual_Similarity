# -*- coding: utf-8 -*-
'''
Created on 2018年8月1日

@author: fanghaishuo
'''
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow import set_random_seed

set_random_seed(2018)


# from keras.models import Model

class SentenceSimilarity():
    def __init__(self, max_len, embedding_size, vocab_size, hidden_size, num_classes, learning_rate,
                 #initializer=tf.glorot_normal_initializer(seed=2018)):
                 initializer=tf.contrib.layers.xavier_initializer()):

        self.max_len = max_len
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.initializer = initializer
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.input_x = tf.placeholder(tf.int32, [None, None], name='input_x')
        self.x_lens = tf.placeholder(tf.int32, [None], name='x_lens')
        self.input_y = tf.placeholder(tf.int32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.batch_size = tf.placeholder(tf.int32, name='batch_size')
        # self.num_classes =tf.placeholder(tf.int32,name='batch_size')

        # self.classes = tf.placeholder(tf.int32,[None],'num_classes')

        self.instantiate_weights()
        self.inference()
        self.loss = self.define_loss()
        self.train_op = self.train()
        self.pred = tf.cast(tf.argmax(self.cosine, axis=1), tf.int32)
        self.eq = tf.equal(self.input_y, self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.pred), tf.float32), name='accuracy')

    def instantiate_weights(self):
        self.Embedding = tf.get_variable('Embedding', [self.vocab_size + 2, self.embedding_size],
                                         initializer=self.initializer)
        self.w_projection = tf.get_variable('W_projection', shape=[self.hidden_size * 2, self.num_classes],
                                            initializer=self.initializer)
        self.b_projection = tf.get_variable('b_projection', shape=[self.num_classes], initializer=self.initializer)

    def inference(self):
        self.embedded_words = tf.nn.embedding_lookup(self.Embedding, self.input_x)

        def get_a_cell(lstm_size, keep_prob):
            #gru_cell = rnn.GRUCell(lstm_size)
            gru_cell = rnn.LSTMCell(lstm_size)
            gru_cell = rnn.DropoutWrapper(gru_cell, output_keep_prob=keep_prob)
            return gru_cell

        # 两层lstm
        gru_fw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.hidden_size, self.dropout_keep_prob) for _ in range(2)])
        gru_bw = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.hidden_size, self.dropout_keep_prob) for _ in range(2)])
        #gru_fw = get_a_cell(self.hidden_size, self.dropout_keep_prob)
        #gru_bw = get_a_cell(self.hidden_size, self.dropout_keep_prob)
        # 双向的：2*[batch_size,sequence_length,hidden_size]
        outputs_rnn, _ = tf.nn.bidirectional_dynamic_rnn(gru_fw, gru_bw, self.embedded_words,
                                                         sequence_length=self.x_lens, dtype=tf.float32)
        # outputs_rnn,_= tf.nn.bidirectional_dynamic_rnn(multi_fw,multi_bw,self.embedded_words,dtype=tf.float32)

        print(outputs_rnn)

        # outputs = list(map(lambda x:x[0],outputs_rnn))
        outputs_rnn = tf.concat(outputs_rnn, axis=2)
        #outputs_rnn = tf.reduce_mean(outputs_rnn,axis=1)

        # attention
        self.attention_size = 256

        with tf.name_scope('attention'):
            attention_w = tf.get_variable('attention_w', shape=[self.hidden_size * 2, self.attention_size],
                                          initializer=self.initializer)
            attention_b = tf.get_variable('attention_b', shape=[self.attention_size], initializer=self.initializer)
            attention_u = tf.get_variable('attention_u', shape=[self.attention_size], initializer=self.initializer)
            v = tf.tanh(tf.tensordot(outputs_rnn, attention_w, axes=[[2], [0]]) + attention_b)
            uv = tf.tensordot(v, attention_u, axes=[[2], [0]])

            alphas = tf.nn.softmax(uv, name='alphas')

        self.last_output = tf.reduce_sum(outputs_rnn * tf.expand_dims(alphas, -1), 1)
        #last_output = outputs_rnn
        self.outputs_rnn = tf.nn.l2_normalize(self.last_output, 1, name='vectors')
        self.outputs_rnn = tf.nn.dropout(self.outputs_rnn,self.dropout_keep_prob)

        self.w_projection = tf.nn.l2_normalize(self.w_projection, 1, name='classes_vector')
        print(self.outputs_rnn)
        # 不需要加bias，向量相乘 [Batch,hidden_size] *[hidden_size,num_class]，相当于和每个类的中心做点乘
        self.cosine = tf.matmul(self.outputs_rnn, self.w_projection)

    def am_softmax_loss(self, y_true, margin=0.3, scale=30):
        y_true = tf.expand_dims(y_true, 1)
        batch_idxs = tf.range(0, self.batch_size)
        batch_idxs = tf.expand_dims(batch_idxs, 1)
        idxs = tf.concat([batch_idxs, y_true], axis=1)
        self.idxs = idxs
        y_true_pred = tf.gather_nd(self.cosine, idxs)  # [Batch,] 取出每一个里面的ture对应的cos值，使得最大
        y_true_pred = tf.expand_dims(y_true_pred, 1)  # [Batch,1]
        self.y_true_pred = y_true_pred  # [Batch,1]
        y_true_pred_margin = y_true_pred - margin
        _Z = tf.concat([self.cosine, y_true_pred_margin], 1)
        _Z = _Z * scale
        logZ = tf.reduce_logsumexp(_Z, 1, keep_dims=True)
        logZ = logZ + tf.log(1 - tf.exp(scale * y_true_pred - logZ))
        return tf.reduce_mean(- y_true_pred_margin * scale + logZ, axis=0)[0]

    def define_loss(self):
        with tf.name_scope("loss"):
            losses = self.am_softmax_loss(self.input_y)
        return losses

    def train(self):
        ##to do:learing_rate decay
        #self.learning_rate = tf.train.exponential_decay(self.learning_rate, global_step=self.global_step,
         #                                              decay_rate=0.99, decay_steps=100)
        train_op = tf.contrib.layers.optimize_loss(self.loss, global_step=self.global_step,
                                                   learning_rate=self.learning_rate, optimizer='Adam')
        return train_op
