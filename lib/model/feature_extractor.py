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
import h5py
from tqdm import tqdm

from utils.cython_bbox import bbox_overlaps
from utils.blob import im_list_to_blob

from model.config import cfg
from model.bbox_transform import clip_boxes, bbox_transform_inv

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

def im_detect(sess, net, im, phrase):
  blobs, im_scales = _get_blobs(im)
  assert len(im_scales) == 1, "Only single-image batch implemented"

  im_blob = blobs['data']
  blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

  _, scores, bbox_pred, rois, fc7, phrases = net.test_image(sess, blobs['data'], blobs['im_info'], phrase)

  boxes = rois[:, 1:5] / im_scales[0]
  out_scores = np.zeros((scores.shape[0], scores.shape[1], 5), np.float32)
  for i, cls_prob, box_deltas in zip(range(scores.shape[0]), scores, bbox_pred):
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      pred_boxes = bbox_transform_inv(boxes, box_deltas)
      pred_boxes = _clip_boxes(pred_boxes, im.shape)
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.REGION_CLASSIFIER == 'entropy':
      cls_scores = cls_prob[:, 1]
    else:
      cls_scores = cls_prob

    out_scores[i] = np.hstack((pred_boxes[:,4:], cls_scores[:, np.newaxis]))

  return out_scores, fc7, phrases

def cache_gt_features(sess, net, imdb, weights_filename, max_per_image=100, thresh=0.):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  num_images = len(imdb.image_index)
  all_boxes = []
  phrase_batches = []
  phrases = imdb.phrases
  roidb = imdb.roidb
  tag = weights_filename.split(os.path.sep)[0]
  dataset, split = imdb.name.split('_')
  output_dir = os.path.join('external', 'cite', 'data', dataset, tag)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  thresh = 0.5
  if imdb._image_set == 'train':
    thresh = 0.6

  #output_dir = os.path.join('/scratch2/bplum/new_cite/referit')
  pairs, overlaps, scores = [], [], []
  phrases = set()
  loc_predictions = []
  for i in tqdm(range(num_images), desc='caching features', total=num_images):
    im = cv2.imread(imdb.image_path_at(i))
    if im is None:
      im = np.zeros((500,300,3))

    roi = roidb[imdb._image_index[i]]
    im_id = imdb._im_ids[imdb._image_index[i]]
    if isinstance(im_id, int):
      im_id = str(im_id)

    phrase_features = roi['vecs']
    if len(phrase_features) < 1:
      continue

    num_gt_annotations = len(roi['phrases'])
    predictions, features, _ = im_detect(sess, net, im, phrase_features)
    np.save(os.path.join(output_dir, im_id + '.npy'), features)
    for p_id, phrase, boxes, gt in zip(range(len(predictions)), roi['processed_phrases'], 
                                       predictions, roi['boxes']):

      gt = gt.reshape((1, 4)).astype(np.float)
      pred = boxes[:, :-1].reshape((-1, 4)).astype(np.float)
      iou = bbox_overlaps(pred, gt).squeeze()
      pos = np.where(iou >= thresh)[0]
      if imdb._image_set == 'train':
        num_pos = len(pos)
        num_neg = num_pos * 2
        negs = np.where(iou < 0.3)[0]
        if len(negs) < num_neg:
          negs = np.where(iou < 0.4)[0]

        pos = (pos, negs, num_neg)

      overlaps.append(pos)
      loc_predictions.append(float(iou[np.argmax(boxes[:, -1])] >= 0.5))
      gt_phrases = set([phrase])
      phrase = phrase.strip().replace(' ', '+')
      phrases.add(phrase)

      pairs.append([im_id, phrase, p_id, int(p_id < num_gt_annotations)])
      scores.append(boxes)

  out_data = {'phrases' : phrases, 'pairs' : pairs, 'overlaps' : overlaps}
  pickle.dump(out_data, open(os.path.join(output_dir, split + '.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
  pickle.dump(scores, open(os.path.join(output_dir, split + '_scores.pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)
  print('created output for {:d} pairs'.format(len(pairs)))
  print('localization', sum(loc_predictions) / len(loc_predictions))

