# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

from model.config import cfg

class PhraseEncoder(object):
  def __init__(self, vecs, max_tokens):
    self._vecs = vecs
    self._max_tokens = max_tokens

  def _phrase_to_tail(self, is_training, reuse=None):

    outdim = self._feature_dim
    if cfg.BBOX_FEATS:
      # box features are 5-D
      outdim += 5

    #if True:
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      l2_reg = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
      word_embeddings = tf.get_variable('word_embeddings', self._vecs.shape, initializer=tf.constant_initializer(self._vecs), trainable = False)
      embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, self._tokens)
      if cfg.TEXT_FEAT_DIM != self._vecs.shape[1]:
        embedded_word_ids = slim.fully_connected(embedded_word_ids, cfg.TEXT_FEAT_DIM, activation_fn = None,
                                                 weights_regularizer = l2_reg,
                                                 reuse=reuse,
                                                 scope = 'word_projections')

      num_words = tf.reduce_sum(tf.to_float(self._tokens > 0), 1, keep_dims=True) + 1e-10
      self._phrase = tf.nn.l2_normalize(tf.reduce_sum(embedded_word_ids, 1) / num_words, 1)
      p1 = slim.fully_connected(self._phrase, outdim, activation_fn = None,
                                weights_regularizer = l2_reg,
                                reuse=reuse,
                                scope = 'phrase_1')
      bn = tf.contrib.layers.batch_norm(p1, decay=0.99, center=True, scale=True,
                                        is_training=is_training,
                                        reuse=reuse,
                                        trainable=True,
                                        updates_collections=None,
                                        scope='bnorm_p1')
      
      bn = tf.nn.relu(bn, 'relu_bnorm_p1')
    return bn
