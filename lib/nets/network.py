# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import os
import pickle
import numpy as np

from nets.phrase_encoder import PhraseEncoder

from layer_utils.snippets import generate_anchors_pre
from layer_utils.proposal_layer import proposal_layer
from layer_utils.proposal_top_layer import proposal_top_layer
from layer_utils.anchor_target_layer import anchor_target_layer
from layer_utils.proposal_target_layer import proposal_target_layer
from layer_utils.layer_regularizers import weight_l2_regularizer

from model.config import cfg, get_output_dir

class Network(PhraseEncoder):
  def __init__(self, output_dir, vecs, max_tokens):
    PhraseEncoder.__init__(self, vecs, max_tokens)
    self._predictions = {}
    self._losses = {}
    self._anchor_targets = {}
    self._proposal_targets = {}
    self._layers = {}
    self._gt_image = None
    self._variables_to_fix = {}
    self._cca_parameters = None
    if output_dir is not None and cfg.CCA_INIT and cfg.REGION_CLASSIFIER != 'cite':
      fn = os.path.join(output_dir, 'cca_parameters.pkl')
      self._cca_parameters = pickle.load(open(fn, 'rb'))

  def _reshape_layer(self, bottom, num_dim, name):
    input_shape = tf.shape(bottom)
    with tf.variable_scope(name) as scope:
      # change the channel to the caffe format
      to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
      # then force it to have channel 2
      reshaped = tf.reshape(to_caffe,
                            tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))
      # then swap the channel back
      to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
      return to_tf

  def _softmax_layer(self, bottom, name):
    if name.startswith('rpn_cls_prob_reshape'):
      input_shape = tf.shape(bottom)
      bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
      reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
      return tf.reshape(reshaped_score, input_shape)
    return tf.nn.softmax(bottom, name=name)

  def _proposal_top_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = proposal_top_layer(
        rpn_cls_prob,
        rpn_bbox_pred,
        self._im_info,
        self._feat_stride,
        self._anchors,
        self._num_anchors
      )
      rois.set_shape([cfg.TEST.RPN_TOP_N, 5])
      rpn_scores.set_shape([cfg.TEST.RPN_TOP_N, 1])

    return rois, rpn_scores

  def _proposal_layer(self, rpn_cls_prob, rpn_bbox_pred, name):
    with tf.variable_scope(name) as scope:
      rois, rpn_scores = proposal_layer(
        rpn_cls_prob,
        rpn_bbox_pred,
        self._im_info,
        self._mode,
        self._feat_stride,
        self._anchors,
        self._num_anchors
      )
      rois.set_shape([None, 5])
      rpn_scores.set_shape([None, 1])

    return rois, rpn_scores

  # Only use it if you have roi_pooling op written in tf.image
  def _roi_pool_layer(self, bootom, rois, name):
    with tf.variable_scope(name) as scope:
      return tf.image.roi_pooling(bootom, rois,
                                  pooled_height=cfg.POOLING_SIZE,
                                  pooled_width=cfg.POOLING_SIZE,
                                  spatial_scale=1. / 16.)[0]

  def _crop_pool_layer(self, bottom, rois, name):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      # Get the normalized coordinates of bounding boxes
      bottom_shape = tf.shape(bottom)
      height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self._feat_stride[0])
      width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self._feat_stride[0])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = cfg.POOLING_SIZE * 2
      crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return slim.max_pool2d(crops, [2, 2], padding='SAME')

  def _dropout_layer(self, bottom, name, ratio=0.5):
    return tf.nn.dropout(bottom, ratio, name=name)

  def _anchor_target_layer(self, rpn_cls_score, name):
    with tf.variable_scope(name) as scope:
      rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(
        anchor_target_layer,
        [rpn_cls_score, self._gt_boxes, self._im_info, self._feat_stride, self._anchors, self._num_anchors],
        [tf.float32, tf.float32, tf.float32, tf.float32],
        name="anchor_target")

      rpn_labels.set_shape([1, 1, None, None])
      rpn_bbox_targets.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_inside_weights.set_shape([1, None, None, self._num_anchors * 4])
      rpn_bbox_outside_weights.set_shape([1, None, None, self._num_anchors * 4])

      rpn_labels = tf.to_int32(rpn_labels, name="to_int32")
      self._anchor_targets['rpn_labels'] = rpn_labels
      self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
      self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
      self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

    return rpn_labels

  def _rpn_proposal_target_layer(self, rois, roi_scores, name):
    with tf.variable_scope(name) as scope:
      rois, roi_scores, _, _, _, _ = tf.py_func(
        proposal_target_layer,
        [rois, self._gt_boxes, self._num_classes, roi_scores],
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        name="rpn_proposal_target")

      rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
      roi_scores.set_shape([cfg.TRAIN.BATCH_SIZE])

      self._proposal_targets['rois'] = rois
      return rois, roi_scores

  def _proposal_target_layer(self, rois, name):
    all_labels = []
    all_bbox_target = []
    all_inside_weights = []
    all_outside_weights = []
    for i in range(self._num_phrases):
      gt_box = tf.expand_dims(self._gt_boxes[i], 0)
      with tf.variable_scope(name) as scope:
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
          proposal_target_layer,
          [rois, gt_box, self._num_classes],
          [tf.float32, tf.float32, tf.float32, tf.float32],
          name="proposal_target")

        labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
        bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
        bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
        bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
        all_labels.append(labels)
        all_bbox_target.append(bbox_targets)
        all_inside_weights.append(bbox_inside_weights)
        all_outside_weights.append(bbox_outside_weights)

    labels = tf.stack(all_labels)
    bbox_targets = tf.stack(all_bbox_target)
    bbox_inside_weights = tf.stack(all_inside_weights)
    bbox_outside_weights = tf.stack(all_outside_weights)
    self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
    self._proposal_targets['bbox_targets'] = bbox_targets
    self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
    self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

  def _anchor_component(self):
    with tf.variable_scope('ANCHOR_' + self._tag) as scope:
      # just to get the shape right
      height = tf.to_int32(tf.ceil(self._im_info[0] / np.float32(self._feat_stride[0])))
      width = tf.to_int32(tf.ceil(self._im_info[1] / np.float32(self._feat_stride[0])))
      anchors, anchor_length = generate_anchors_pre(
        height,
        width,
        self._feat_stride,
        self._anchor_scales,
        self._anchor_ratios
      )
      anchors.set_shape([None, 4])
      anchor_length.set_shape([])
      self._anchors = anchors
      self._anchor_length = anchor_length

  def _build_roi_features(self, is_training=True):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    net_conv = self._image_to_head(is_training)

    with tf.variable_scope(self._scope, self._scope):
      # build the anchors for the image
      self._anchor_component()
      # region proposal network
      rois = self._region_proposal(net_conv, is_training, initializer)
      # region of interest pooling
      if cfg.POOLING_MODE == 'crop':
        pool5 = self._crop_pool_layer(net_conv, rois, "pool5")
      else:
        raise NotImplementedError

    box_feats = None
    if cfg.BBOX_FEATS:
      x1 = rois[:, 1] / self._im_info[1]
      y1 = rois[:, 2] / self._im_info[0]
      x2 = rois[:, 3] / self._im_info[1]
      y2 = rois[:, 4] / self._im_info[0]
      box_area = (x2 - x1) * (y2 - y1)
      area = box_area / (self._im_info[0] * self._im_info[1])
      box_feats = tf.stack([x1, y1, x2, y2, area], axis=1)

    return pool5, rois, box_feats


  def _build_network(self, is_training=True, reuse=None):
    # select initializers
    if cfg.TRAIN.TRUNCATED:
      initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
    else:
      initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
      initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

    phrase_embed = self._phrase_to_tail(is_training, reuse)
    fc7 = self._predictions["fc7"]
    with tf.variable_scope(self._scope, self._scope, reuse=reuse):
      # region classification
      cls_prob, bbox_pred = self._region_classification(fc7, phrase_embed, is_training, 
                                                        initializer, initializer_bbox, reuse)

    return cls_prob, bbox_pred

  def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = tf.abs(in_box_diff)
    smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. / sigma_2)))
    in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = tf.reduce_mean(tf.reduce_sum(
      out_loss_box,
      axis=dim
    ))
    return loss_box

  def _add_rpn_losses(self, sigma_rpn=3.0):
    with tf.variable_scope('LOSS_RPN_' + self._tag) as scope:
      # RPN, class loss
      rpn_cls_score = tf.reshape(self._predictions['rpn_cls_score_reshape'], [-1, 2])
      rpn_label = tf.reshape(self._anchor_targets['rpn_labels'], [-1])
      rpn_select = tf.where(tf.not_equal(rpn_label, -1))
      rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
      rpn_label = tf.reshape(tf.gather(rpn_label, rpn_select), [-1])
      rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score, labels=rpn_label))

      # RPN, bbox loss
      rpn_bbox_pred = self._predictions['rpn_bbox_pred']
      rpn_bbox_targets = self._anchor_targets['rpn_bbox_targets']
      rpn_bbox_inside_weights = self._anchor_targets['rpn_bbox_inside_weights']
      rpn_bbox_outside_weights = self._anchor_targets['rpn_bbox_outside_weights']
      rpn_loss_box = self._smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                          rpn_bbox_outside_weights, sigma=sigma_rpn, dim=[1, 2, 3])

      self._losses['rpn_cross_entropy'] = rpn_cross_entropy
      self._losses['rpn_loss_box'] = rpn_loss_box

  def _get_cls_losses(self):
    with tf.variable_scope('LOSS_CLS_' + self._tag) as scope:
      # RCNN, class loss
      cls_score = self._predictions["cls_score"]
      if cfg.REGION_CLASSIFIER == 'entropy':
        cls_score = tf.reshape(cls_score, [-1, self._num_classes])
        label = tf.reshape(self._proposal_targets["labels"], [-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))
      elif cfg.REGION_CLASSIFIER == 'cite':
        labels = tf.to_float(tf.squeeze(self._proposal_targets["labels"]) > 0)
        pos_score = tf.reshape(tf.boolean_mask(cls_score, labels > 0), [-1])
        cross_entropy = tf.reduce_sum(tf.log(1+tf.exp(-pos_score)))
        neg_score = tf.reshape(tf.boolean_mask(cls_score, labels < 1), [-1])

        # always sample at least one negative
        num_neg_samples = tf.maximum(cfg.TRAIN.CITE_NEG_TO_POS_RATIO * tf.shape(pos_score)[0], 1)
        ind = tf.random_shuffle(tf.range(tf.shape(neg_score)[0]))[:num_neg_samples]
        neg_score = tf.gather(neg_score, ind)

        # negatives get a label of -1, so its scores are positive in the function
        cross_entropy += tf.reduce_sum(tf.log(1+tf.exp(neg_score)))
        cross_entropy /= (tf.to_float(tf.shape(neg_score)[0] + tf.shape(pos_score)[0]) + 1e-10)
        concept_weight = self._predictions["concept_weights"]
        concept_loss = tf.reduce_sum(tf.norm(concept_weight, axis=1, ord=1)) / self._num_phrases
        cross_entropy += concept_loss * cfg.TRAIN.CITE_CONCEPT_LOSS_WEIGHT
      elif cfg.REGION_CLASSIFIER == 'embed':
        cls_score = tf.reshape(cls_score, [-1])
        if cfg.TRAIN.EMBED_SINGLE_CATEGORY:
          # All positive phrase-region pairs should be separated by a 
          # margin with all negative phrase-region pairs (even for
          # different phrases)
          labels = tf.to_float(tf.squeeze(self._proposal_targets["labels"]) > 0)
          pos_scores = tf.reshape(tf.boolean_mask(cls_score, labels > 0), [-1, 1])
          neg_scores = tf.reshape(tf.boolean_mask(cls_score, labels < 1), [1, -1])
          cross_entropy = tf.clip_by_value(cfg.TRAIN.EMBED_MARGIN + neg_scores - pos_scores, 0, 1e6)
          num_samples = cfg.TRAIN.EMBED_TOPK_VIOLATIONS * self._num_phrases
          cross_entropy = tf.reduce_mean(tf.nn.top_k(cross_entropy, k=num_samples)[0])
        else:
          # Loss as described in Wang et al. "Learning Two-Branch Neural 
          # Networks for Image-Text Matching Tasks." TPAMI, 2019.
          labels = tf.to_float(tf.squeeze(self._proposal_targets["labels"]) > 0)
          pos_score = cls_score * labels
          neg_score = (cfg.TRAIN.EMBED_MARGIN + cls_score) * tf.to_float(labels < 1)
          
          # loss for lambda1 from paper
          phrase_labels = tf.expand_dims(self._phrase_labels, 2)
          phrase_region_pos = tf.expand_dims(pos_score, 0) * phrase_labels
          phrase_region_neg = tf.expand_dims(neg_score, 1) * tf.to_float(phrase_labels < 1)
          phrase_phrase_region_score = phrase_region_neg - phrase_region_pos
          
          # check for locations which have high enough overlap with the gt
          good_locations = tf.to_float(tf.reduce_sum(labels, axis=0) > 0)
          phrase_phrase_region_score = tf.reshape(phrase_phrase_region_score, [-1, tf.shape(good_locations)[0]])
          phrase_phrase_region_score = tf.transpose(phrase_phrase_region_score)
          phrase_phrase_region_score = tf.nn.top_k(phrase_phrase_region_score, k=cfg.TRAIN.EMBED_TOPK_VIOLATIONS)[0]
          phrase_phrase_region_score *= tf.expand_dims(good_locations, 1)
          phrase_phrase_region_score = tf.clip_by_value(phrase_phrase_region_score, 0, 1e6)
          num_good_locations = tf.reduce_sum(good_locations) + 1e-10
          phrase_phrase_region_loss = tf.reduce_sum(phrase_phrase_region_score) / num_good_locations
          weighted_phrase_phrase_region = phrase_phrase_region_loss * cfg.TRAIN.EMBED_PHRASE_PHRASE_REGION_LOSS
          cross_entropy = weighted_phrase_phrase_region

          # loss for lambda2 from the paper
          phrase_region_region_score = tf.expand_dims(neg_score, 2) - tf.expand_dims(pos_score, 1)
          phrase_region_region_score = tf.matrix_band_part(phrase_region_region_score, 0, -1)
          phrase_region_region_score = tf.reshape(phrase_region_region_score, [self._num_phrases, -1])
          phrase_region_region_score = tf.nn.top_k(phrase_region_region_score, k=cfg.TRAIN.EMBED_TOPK_VIOLATIONS)[0]
          phrase_region_region_loss = tf.reduce_mean(tf.clip_by_value(phrase_region_region_score, 0, 1e6))
          weighted_phrase_region_region = phrase_region_region_loss * cfg.TRAIN.EMBED_PHRASE_REGION_REGION_LOSS
          cross_entropy += weighted_phrase_region_region        

        # neighborhood constraint, lambda3
        labels = tf.expand_dims(labels, 1)
        region_score = tf.expand_dims(self._predictions["region_score"], 0)
        pos_regions = region_score * labels
        neg_regions = (cfg.TRAIN.EMBED_MARGIN + region_score) * tf.to_float(labels < 1)
        region_differences = tf.matrix_band_part(neg_regions - pos_regions, 0, -1)
        region_differences = tf.reshape(region_differences, [self._num_phrases, -1])
        region_differences = tf.nn.top_k(region_differences, k=cfg.TRAIN.EMBED_TOPK_VIOLATIONS)[0]
        region_region_loss = tf.reduce_mean(tf.clip_by_value(region_differences, 0, 1e6))
        weighted_region_loss = region_region_loss * cfg.TRAIN.EMBED_REGION_LOSS
        cross_entropy += weighted_region_loss
          
        # neighborhood constraint, lambda4
        phrase_score = self._predictions["phrase_score"]
        neg_phrase = (cfg.TRAIN.EMBED_MARGIN + phrase_score) * tf.to_float(self._phrase_labels < 1)
        pos_phrase = phrase_score * tf.to_float(self._phrase_labels > 0)
        
        # just get most violated constraint because topk might dilute the effect
        # of the phrase loss because there aren't a lot of positive pairs
        phrase_differences = tf.reduce_max(neg_phrase - pos_phrase, axis=1)
        phrase_phrase_loss = tf.reduce_mean(tf.clip_by_value(phrase_differences, 0, 1e6))
        weighted_phrase_loss = phrase_phrase_loss * cfg.TRAIN.EMBED_PHRASE_LOSS
        cross_entropy += weighted_phrase_loss
      elif cfg.REGION_CLASSIFIER == 'classifier':
        label = tf.to_float(tf.squeeze(self._proposal_targets["labels"]))
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cls_score, labels=label))

      # RCNN, bbox loss
      bbox_pred = self._predictions['bbox_pred']
      bbox_targets = self._proposal_targets['bbox_targets']
      bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
      bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
      loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, dim=[2])

    return cross_entropy, loss_box

  def _add_losses(self, cross_entropy, loss_box):
    with tf.variable_scope('LOSS_' + self._tag) as scope:
      rpn_cross_entropy = self._losses['rpn_cross_entropy']
      rpn_loss_box = self._losses['rpn_loss_box']
      loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

      regularization_loss = tf.add_n(tf.losses.get_regularization_losses(), 'regu')

      self._losses['cross_entropy'] = cross_entropy
      self._losses['loss_box'] = loss_box
      self._losses['total_loss'] = loss + regularization_loss

    return loss

  def _region_proposal(self, net_conv, is_training, initializer):
    rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training, weights_initializer=initializer,
                        scope="rpn_conv/3x3")
    rpn_cls_score = slim.conv2d(rpn, self._num_anchors * 2, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_cls_score')
    # change it so that the score has 2 as its channel size
    rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score, 2, 'rpn_cls_score_reshape')
    rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")
    rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")
    rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, self._num_anchors * 2, "rpn_cls_prob")
    rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1], trainable=is_training,
                                weights_initializer=initializer,
                                padding='VALID', activation_fn=None, scope='rpn_bbox_pred')
    if is_training:
      rois, roi_scores = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      rpn_labels = self._anchor_target_layer(rpn_cls_score, "anchor")
      # Try to have a deterministic order for the computing graph, for reproducibility
      with tf.control_dependencies([rpn_labels]):
        rois, _ = self._rpn_proposal_target_layer(rois, roi_scores, "rpn_rois")
    else:
      if cfg.TEST.MODE == 'nms':
        rois, _ = self._proposal_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      elif cfg.TEST.MODE == 'top':
        rois, _ = self._proposal_top_layer(rpn_cls_prob, rpn_bbox_pred, "rois")
      else:
        raise NotImplementedError

    self._predictions["rpn_cls_score"] = rpn_cls_score
    self._predictions["rpn_cls_score_reshape"] = rpn_cls_score_reshape
    self._predictions["rpn_cls_prob"] = rpn_cls_prob
    self._predictions["rpn_cls_pred"] = rpn_cls_pred
    self._predictions["rpn_bbox_pred"] = rpn_bbox_pred
    self._predictions["rois"] = rois

    return rois

  def _region_classification(self, fc7, phrase_embed, is_training, initializer, initializer_bbox, reuse):
    joint_embed = fc7 * tf.expand_dims(phrase_embed, 1)
    if cfg.REGION_CLASSIFIER == 'entropy' or cfg.REGION_CLASSIFIER == 'cite':
      cls_score = slim.fully_connected(joint_embed, self._num_classes,
                                       weights_initializer=initializer,
                                       trainable=is_training,
                                       reuse=reuse,
                                       activation_fn=None, scope='cls_score_2')
      cls_prob = self._softmax_layer(cls_score, "cls_prob")
    elif cfg.REGION_CLASSIFIER == 'embed' or cfg.REGION_CLASSIFIER == 'cite':
      phrase, region = self._phrase, fc7
      for i, outdim in enumerate(cfg.EMBED_LAYERS):
        if outdim  < 1:
          outdim = self._feature_dim

        scaling = 1
        weights_init = None
        cca_mean = 0
        vis_cca_mean = 0
        vis_weights_init = None
        vis_reg, lang_reg = None, None
        activation_fn = None
        if cfg.CCA_INIT and self._cca_parameters is not None:
          cca_mean, cca_proj = self._cca_parameters[i]['lang_mean'], self._cca_parameters[i]['lang_proj']
          weights_init = tf.constant_initializer(cca_proj, dtype=tf.float32)
          lang_reg = weight_l2_regularizer(self._cca_parameters[i]['lang_proj'], cfg.TRAIN.CCA_L2_REG)
          scaling = self._cca_parameters[i]['scaling']
          outdim = len(scaling)
          vis_cca_mean, vis_cca_proj = self._cca_parameters[i]['vis_mean'], self._cca_parameters[i]['vis_proj']
          vis_weights_init = tf.constant_initializer(vis_cca_proj, dtype=tf.float32)
          vis_reg = weight_l2_regularizer(self._cca_parameters[i]['vis_proj'], cfg.TRAIN.CCA_L2_REG)
        elif cfg.BBOX_FEATS and i == 0:
          # they're 5-D features
          outdim += 5

        # add ReLUs between layers (CCA should be trained with ReLUs as well)
        if (i + 1) < len(cfg.EMBED_LAYERS):
          activation_fn = tf.nn.relu

        phrase = slim.fully_connected(phrase - cca_mean, outdim, activation_fn=activation_fn,
                                      trainable=is_training,
                                      weights_regularizer = lang_reg,
                                      biases_regularizer = tf.contrib.layers.l1_regularizer(cfg.TRAIN.WEIGHT_DECAY),
                                      weights_initializer = weights_init,
                                      scope = 'cls_phrase_' + str(i)) * scaling
        region = slim.fully_connected(region - vis_cca_mean, outdim, activation_fn=activation_fn,
                                      trainable=is_training,
                                      weights_regularizer = vis_reg,
                                      biases_regularizer = tf.contrib.layers.l1_regularizer(cfg.TRAIN.WEIGHT_DECAY),
                                      weights_initializer = vis_weights_init,
                                      scope = 'cls_region_' + str(i)) * scaling

      normed_region = tf.nn.l2_normalize(region, 1)
      normed_phrase = tf.nn.l2_normalize(phrase, 1)
      region_phrase_embedding = normed_region * tf.expand_dims(normed_phrase, 1)
      if cfg.REGION_CLASSIFIER == 'embed':
        cls_score = tf.reduce_sum(region_phrase_embedding, 2)
        cls_prob = cls_score
        # used for neighborhood constraints
        self._predictions["region_score"] = tf.reduce_sum(tf.expand_dims(normed_region, 1) * tf.expand_dims(normed_region, 0), 2)
        self._predictions["phrase_score"] = tf.reduce_sum(tf.expand_dims(normed_phrase, 1) * tf.expand_dims(normed_phrase, 0), 2)
      else:
        concept_weights = self._cite_concept_weights(outdim, is_training)
        l2_reg = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        joint_fc1 = slim.fully_connected(region_phrase_embedding, outdim, activation_fn = None,
                                         weights_regularizer = l2_reg,
                                         scope = 'joint_fc1')
        joint_bn1 = tf.nn.relu(slim.batch_norm(joint_fc1, decay=0.99, center=True, scale=True,
                                              is_training=is_training,
                                              trainable=is_training,
                                              updates_collections=None,
                                              scope='joint_fc1/bnorm'))
        concept_embed = None
        for concept_id in range(cfg.TRAIN.CITE_NUM_CONCEPTS):
          joint_fc2 = slim.fully_connected(joint_bn1, cfg.TRAIN.CITE_FINAL_EMBED, activation_fn = None,
                                           weights_regularizer = l2_reg,
                                           scope = 'joint_concept%i_fc2' % concept_id)
          joint_bn2 = tf.nn.relu(slim.batch_norm(joint_fc2, decay=0.99, center=True, scale=True,
                                                 is_training=is_training,
                                                 trainable=is_training,
                                                 updates_collections=None,
                                                 scope='joint_concept%i_fc2/bnorm' % concept_id))
          weighted_concept = joint_bn2 * tf.reshape(concept_weights[:, concept_id], [self._num_phrases, 1, 1])
          if concept_embed == None:
            concept_embed = weighted_concept
          else:
            concept_embed += weighted_concept

        assert(concept_embed is not None)
        cls_score = slim.fully_connected(concept_embed, 1, activation_fn = None,
                                         weights_regularizer = l2_reg,
                                         scope = 'phrase_region_score')
        cls_score = tf.squeeze(cls_score, [2])
        cls_prob = 1. / (1. + tf.exp(-cls_score))      
    elif cfg.REGION_CLASSIFIER == 'classifier':
      cls_score = tf.reduce_sum(joint_embed, 2)
      cls_prob = tf.nn.sigmoid(cls_score)
    else:
      raise NotImplementedError

    cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
    bbox_pred = slim.fully_connected(joint_embed, self._num_classes * 4, 
                                     weights_initializer=initializer_bbox,
                                     trainable=is_training,
                                     reuse=reuse,
                                     activation_fn=None, scope='bbox_pred_2')

    self._predictions["cls_score"] = cls_score
    self._predictions["cls_pred"] = cls_pred
    self._predictions["cls_prob"] = cls_prob
    self._predictions["bbox_pred"] = bbox_pred

    return cls_prob, bbox_pred

  def _cite_concept_weights(self, outdim, is_training):
    l2_reg = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    concept_fc1 = slim.fully_connected(self._phrase, outdim, activation_fn = None,
                                       weights_regularizer = l2_reg,
                                       scope = 'concept_fc1')
    concept_bn = tf.nn.relu(slim.batch_norm(concept_fc1, decay=0.99, center=True, scale=True,
                                            is_training=is_training,
                                            trainable=is_training,
                                            updates_collections=None,
                                            scope='concept_fc1/bnorm'))
    concept_weights = slim.fully_connected(concept_fc1, cfg.TRAIN.CITE_NUM_CONCEPTS, activation_fn = None,
                                           weights_regularizer = l2_reg,
                                           scope = 'concept_fc2')
    self._predictions["concept_weights"] = concept_weights
    concept_weights = tf.nn.softmax(concept_weights)
    return concept_weights

  def _image_to_head(self, is_training, reuse=None):
    raise NotImplementedError

  def _head_to_tail(self, pool5, is_training, reuse=None):
    raise NotImplementedError

  def extract_scores(self, sess, fc7, phrases, phrase_scores):
    feed_dict = {self._predictions["fc7_input"]: fc7,
                 self._phrase_input : phrases,
                 self._num_phrases: len(phrases),
                 self._num_boxes: len(fc7)}
    if phrase_scores is not None:
      feed_dict[self._best_index] = phrase_scores[:, 1]
  
    rois, feat, index = sess.run([self._predictions['bbox_pred'],
                                  self._predictions['cls_prob'],
                                  self._predictions["best_index"]], feed_dict=feed_dict)

    if phrase_scores is not None:
      feat = phrase_scores[:, 0]

    return rois, feat, index

  def create_phrase_extractor(self, mode, num_classes, tag=None):
    training = mode == 'TRAIN'
    im_features = self._feature_dim
    if cfg.BBOX_FEATS:
      # box features are 5-D
      im_features += 5

    self._num_phrases = tf.placeholder(tf.int32)
    self._num_boxes = tf.placeholder(tf.int32)
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._phrase_input = tf.placeholder(tf.int32, shape=[None, self._max_tokens])
    self._predictions["fc7_input"] = tf.placeholder(tf.float32, shape=[None, im_features])
    self._tag = tag

    if cfg.REGION_CLASSIFIER == 'cite':
      self._best_index = tf.reshape(tf.placeholder(tf.int32, shape=[None]), [self._num_phrases])

    self._tokens = tf.reshape(self._phrase_input, [self._num_phrases, self._max_tokens])
    self._predictions["fc7"] = tf.reshape(self._predictions["fc7_input"], [self._num_boxes, im_features])

    self._num_classes = num_classes
    self._mode = mode
    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer,
                    biases_initializer=tf.constant_initializer(0.0)):
      cls_prob, bbox_pred = self._build_network(training)
      stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
      means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
      bbox_pred *= stds
      bbox_pred += means
      self._predictions["bbox_pred"] = bbox_pred
      boxes = bbox_pred[:, :, 4:]

      if cfg.REGION_CLASSIFIER == 'entropy':
        score = cls_prob[:, :, 1]
      else:
        score = cls_prob

      if cfg.REGION_CLASSIFIER == 'cite':
        best_index = self._best_index
      else:
        score, best_index =  tf.nn.top_k(score, k=cfg.TOP_K_PER_PHRASE)
        best_index = tf.reshape(best_index, [-1])

      ind = tf.expand_dims(tf.range(self._num_phrases), 1)
      ind = tf.reshape(tf.tile(ind, [1, cfg.TOP_K_PER_PHRASE]), [-1])
      ind = tf.stack([ind, best_index], axis=1)
      if cfg.REGION_CLASSIFIER == 'cite':
        score = tf.gather_nd(score, ind)

      self._predictions["best_index"] = tf.reshape(best_index, [-1, cfg.TOP_K_PER_PHRASE])
      if training:
        self._predictions["bbox_pred"] = tf.gather_nd(boxes, ind)
      self._predictions['cls_prob'] = tf.reshape(score, [-1, cfg.TOP_K_PER_PHRASE])
      return self._predictions["bbox_pred"], self._predictions['cls_prob'], self._predictions["best_index"]

  def extract_features(self, sess, image, im_info):
    feed_dict = {self._image: image,
                 self._im_info: im_info}
    rois, feat = sess.run([self._predictions['rois'],
                           self._predictions['fc7']], feed_dict=feed_dict)
    return rois, feat

  def create_feat_extractor(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    training = mode == 'TRAIN'
    self._num_phrases = cfg.TEST.MAX_PHRASES
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._tag = tag

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios
    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer,
                    biases_initializer=tf.constant_initializer(0.0)):

      pool5, rois, box_feats = self._build_roi_features(training)
      fc7 = self._head_to_tail(pool5, training)
      if cfg.BBOX_FEATS:
        fc7 = tf.concat([fc7, box_feats], axis=1)

      self._predictions["fc7"] = fc7

    return rois, fc7

  def create_architecture(self, mode, num_classes, tag=None,
                          anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    training = mode == 'TRAIN'
    testing = mode == 'TEST'
    if training:
      num_phrases = cfg.TRAIN.MAX_PHRASES
    else:
      num_phrases = tf.placeholder(tf.int32)

    self._num_phrases = num_phrases
    self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    self._im_info = tf.placeholder(tf.float32, shape=[3])
    self._gt_boxes = tf.reshape(tf.placeholder(tf.float32, shape=[None, 5]), [num_phrases, 5])
    self._tokens = tf.reshape(tf.placeholder(tf.int32, shape=[None, self._max_tokens]), [num_phrases, self._max_tokens])
    self._num_gt = tf.placeholder(tf.int32)
    self._tag = tag
    if training and cfg.REGION_CLASSIFIER == 'embed':
      self._phrase_labels = tf.placeholder(tf.float32, shape=[num_phrases, num_phrases])

    self._num_classes = num_classes
    self._mode = mode
    self._anchor_scales = anchor_scales
    self._num_scales = len(anchor_scales)

    self._anchor_ratios = anchor_ratios
    self._num_ratios = len(anchor_ratios)

    self._num_anchors = self._num_scales * self._num_ratios

    assert tag != None

    # handle most of the regularizers here
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
    if cfg.TRAIN.BIAS_DECAY:
      biases_regularizer = weights_regularizer
    else:
      biases_regularizer = tf.no_regularizer

    # list as many types of layers as possible, even if they are not used now
    with arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                    slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected], 
                    weights_regularizer=weights_regularizer,
                    biases_regularizer=biases_regularizer, 
                    biases_initializer=tf.constant_initializer(0.0)): 
      pool5, rois, box_feats = self._build_roi_features(training)
      if not testing:
        self._add_rpn_losses()

      fc7 = self._head_to_tail(pool5, training)
      if cfg.BBOX_FEATS:
        fc7 = tf.concat([fc7, box_feats], axis=1)

      self._predictions["fc7"] = fc7
      layers_to_output = {'rois': rois}
      cls_prob, bbox_pred = self._build_network(training)
      if testing:
          stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (self._num_classes))
          means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (self._num_classes))
          bbox_pred *= stds
          bbox_pred += means
          self._predictions["bbox_pred"] = bbox_pred
      else:
        self._proposal_target_layer(rois, "bbox_rois")
        cross_entropy, loss_box = self._get_cls_losses()
        self._add_losses(cross_entropy, loss_box)
        layers_to_output.update(self._losses)

    layers_to_output.update(self._predictions)

    return layers_to_output

  def get_variables_to_restore(self, variables, var_keep_dic):
    raise NotImplementedError

  def fix_variables(self, sess, pretrained_model):
    raise NotImplementedError

  # Extract the head feature maps, for example for vgg16 it is conv5_3
  # only useful during testing mode
  def extract_head(self, sess, image):
    feed_dict = {self._image: image}
    feat = sess.run(self._layers["head"], feed_dict=feed_dict)
    return feat

  # only useful during testing mode
  def test_image(self, sess, image, im_info, phrase):
    num_phrases = min(len(phrase), cfg.TEST.MAX_PHRASES)
    feed_dict = {self._image: image,
                 self._im_info: im_info,
                 self._tokens : phrase,
                 self._num_phrases: num_phrases}

    cls_score, cls_prob, bbox_pred, rois, fc7, phrases = sess.run([self._predictions["cls_score"],
                                                                   self._predictions['cls_prob'],
                                                                   self._predictions['bbox_pred'],
                                                                   self._predictions['rois'],
                                                                   self._predictions['fc7'],
                                                                   self._phrase],
                                                         feed_dict=feed_dict)

    return cls_score, cls_prob, bbox_pred, rois, fc7, phrases

  def train_step(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._tokens : blobs['phrase_feats'], self._num_gt : blobs['n_phrase']}
    if cfg.REGION_CLASSIFIER == 'embed':
      feed_dict[self._phrase_labels] = blobs['phrase_labels']
    rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss, _ = sess.run([self._losses["rpn_cross_entropy"],
                                                                        self._losses['rpn_loss_box'],
                                                                        self._losses['cross_entropy'],
                                                                        self._losses['loss_box'],
                                                                        self._losses['total_loss'],
                                                                        train_op],
                                                                       feed_dict=feed_dict)
    return rpn_loss_cls, rpn_loss_box, loss_cls, loss_box, loss

  def train_step_no_return(self, sess, blobs, train_op):
    feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                 self._gt_boxes: blobs['gt_boxes'], self._tokens : blobs['phrase_feats']}
    if cfg.REGION_CLASSIFIER == 'embed':
      feed_dict[self._phrase_labels] = blobs['phrase_labels']
    sess.run([train_op], feed_dict=feed_dict)


