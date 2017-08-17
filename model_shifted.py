#coding:utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import data_utils
import seq2seq_model
import os
from utils import bw_config,fw_config
#from config import global_memory,global_output

#global global_memory,global_output
def copy_params(session,batch_size=None):
    bw_model = seq2seq_model.Seq2SeqModel(
        vocab_size=bw_config.vocab_size,
        embedding_dim=bw_config.embedding_dim,
        buckets=bw_config.BUCKETS,
        size=bw_config.size,
        num_layers=bw_config.num_layers,
        max_gradient_norm=bw_config.max_gradient_norm,
        batch_size=bw_config.batch_size if not batch_size else batch_size,
        learning_rate=bw_config.learning_rate,
        learning_rate_decay_factor=bw_config.learning_rate_decay_factor,
        use_lstm=True,
        forward_only=True,name='bw_model')
    session.run(tf.global_variables_initializer())
    for ele in bw_model.variables:
        f = open('bw_model/tmp/%s.pkl'%ele.name[9:].replace('/','_'))
        value = pickle.load(f)
        f.close()
        session.run(tf.assign(ele,value))
    bw_model.saver.save(session,'bw/model.ckpt')


if __name__=='__main__':
  with tf.Session() as sess:
    copy_params(sess)
