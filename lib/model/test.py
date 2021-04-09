# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle
import os
import math
from tqdm import tqdm
from collections import Counter

from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def _get_blobs(im):
  """Convert an image and RoIs within that image into network inputs."""
  blobs = {}
  blobs['data'], im_scale_factors = _get_image_blob(im)

  return blobs, im_scale_factors

def _clip_boxes(boxes, im_shape):
  """Clip boxes to image boundaries."""
  # x1 >= 0
  boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
  # y1 >= 0
  boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
  # x2 < im_shape[1]
  boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
  # y2 < im_shape[0]
  boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
  return boxes

def _rescale_boxes(boxes, inds, scales):
  """Rescale boxes according to image rescaling."""
  for i in range(boxes.shape[0]):
    boxes[i,:] = boxes[i,:] / scales[int(inds[i])]

  return boxes

def im_detect(sess, net, boxes, features, phrase, im_shape, phrase_scores):
  bbox_pred, scores, index = net.extract_scores(sess, features, phrase, phrase_scores)
  if cfg.TEST.BBOX_REG:
    # Apply bounding-box regression deltas
    oracle_boxes = []
    pred_boxes = []
    for ind, pred in zip(index, bbox_pred):
      trans = bbox_transform_inv(boxes, pred[:, 4:])
      trans = _clip_boxes(trans, im_shape)
      oracle_boxes.append(np.expand_dims(trans, 0))
      pred_boxes.append(np.expand_dims(trans[ind], 0))

    oracle_boxes = np.vstack(oracle_boxes)
    pred_boxes = np.vstack(pred_boxes)
  else:
    pred_boxes = boxes

  out_scores = np.concatenate((pred_boxes, np.expand_dims(scores, 2)), axis=2)
  return np.reshape(out_scores, [len(phrase), cfg.TOP_K_PER_PHRASE, 5]), oracle_boxes

def cache_im_features(sess, net, im):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)
  rois, fc7 = net.extract_features(sess, im_blob, blobs['im_info'])
  boxes = rois[:, 1:5] / im_scales[0]
  return boxes, fc7, im.shape

def get_features(sess, net, imdb):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  all_boxes = []
  all_features = []
  all_shape = []
  for i in tqdm(range(num_images), desc='caching region features'):
    im = cv2.imread(imdb.image_path_at(i))
    if im is None:
      im = np.zeros((500,300,3))

    rois, features, im_shape = cache_im_features(sess, net, im)

    all_boxes.append(rois)
    all_features.append(features)
    all_shape.append(im_shape)
    
  return all_boxes, all_features, all_shape

def test_net(sess, net, imdb, all_rois, all_features, im_shapes, weights_filename, phrase_scores = None):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  all_boxes = []
  phrase_batches = []
  phrases = imdb.phrases
  roidb = imdb.roidb
  if cfg.REGION_CLASSIFIER == 'cite':
    max_phrases = cfg.TEST.CITE_CLASSIFIER_MAX_PHRASES
  else:
    max_phrases = cfg.TEST.MAX_PHRASES

  n_batches = int(np.ceil(len(phrases) / float(max_phrases)))
  all_ap = np.zeros(len(phrases), np.float32)
  all_phrase_counts = Counter()
  all_top1acc = np.zeros(imdb.num_phrases, np.int32)
  all_total_aug = np.zeros_like(all_top1acc)
  all_top1acc_aug = np.zeros_like(all_top1acc)
  all_oracle = np.zeros(imdb.num_phrases, np.int32)
  all_oracle_aug = np.zeros_like(all_top1acc)
  for batch_id in range(n_batches):
    phrase_start = batch_id*max_phrases
    phrase_end = min((batch_id+1)*max_phrases, len(phrases))
    phrase = np.vstack([imdb.get_word_embedding(p) for p in phrases[phrase_start:phrase_end]])
    all_boxes = []
    oracle_boxes = []
    for i, rois, features, im_shape in tqdm(zip(range(num_images), all_rois, all_features, im_shapes),
                                            desc='scoring/bbreg phrase batch [%i/%i]' %
                                                  (batch_id + 1, n_batches),
                                            total=num_images):
      im_scores = None
      if phrase_scores is not None:
        im_scores = phrase_scores[i, phrase_start:phrase_end, :]

      im_boxes, boxes = im_detect(sess, net, rois, features, phrase, im_shape, im_scores)
      oracle_boxes.append(boxes)
      all_boxes.append(im_boxes)

    output_dir = get_output_dir(imdb, weights_filename)
    det_file = os.path.join(output_dir, 'detections.pkl')
    #with open(det_file, 'rb') as f:
    #  all_boxes = pickle.load(f)
    #with open(det_file, 'wb') as f:
    #  pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('calculating ap for batch')
    ap, phrase_counts, top1acc, total_aug, top1acc_aug, top1acc_oracle, top1acc_aug_oracle = imdb.get_ap(all_boxes, oracle_boxes, phrase_start, phrase_end, output_dir)
    all_ap += ap
    all_phrase_counts += phrase_counts
    all_top1acc += top1acc
    all_total_aug += total_aug
    all_top1acc_aug += top1acc_aug
    all_oracle += top1acc_oracle
    all_oracle_aug += top1acc_aug_oracle

  print('Organizing output')
  imdb.evaluate_detections(all_ap, all_phrase_counts, all_top1acc, all_total_aug, all_top1acc_aug, all_oracle, all_oracle_aug)
