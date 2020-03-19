# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Ross Girshick,
# and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
from model.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
  """Given a roidb, construct a minibatch sampled from it."""
  num_images = len(roidb)
  # Sample random scales to use for each image in this batch
  random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=num_images)
  assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
    'num_images ({}) must divide BATCH_SIZE ({})'. \
    format(num_images, cfg.TRAIN.BATCH_SIZE)

  if cfg.TRAIN.MAX_PHRASES < 1:
    max_phrases = max([len(f['processed_phrases']) for f in roidb])
  else:
    max_phrases = cfg.TRAIN.MAX_PHRASES

  # Get the input image blob, formatted for caffe
  # gt boxes: (x1, y1, x2, y2, cls)
  gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
  npr.shuffle(gt_inds)
  
  gt_inds = gt_inds[:max_phrases]
  im_blob, im_scales, phrase_feats, phrases = _get_image_blob(roidb, random_scale_inds, max_phrases, gt_inds)
  blobs = {'data': im_blob, 'phrase_feats' : phrase_feats, 'n_phrase' : phrases[0] + 1e-10}
  if cfg.REGION_CLASSIFIER == 'embed':
    blobs['phrase_labels'] = np.eye(max_phrases, dtype=np.float32)

  assert len(im_scales) == 1, "Single batch only"
  assert len(roidb) == 1, "Single batch only"  
  gt_inds = gt_inds[:max_phrases]
  gt_boxes = np.zeros((max_phrases, 5), dtype=np.float32)
  gt_boxes[:len(gt_inds), 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
  gt_boxes[:len(gt_inds), 4] = roidb[0]['gt_classes'][gt_inds]
  blobs['gt_boxes'] = gt_boxes
  blobs['im_info'] = np.array(
    [im_blob.shape[1], im_blob.shape[2], im_scales[0]],
    dtype=np.float32)

  return blobs

def _get_image_blob(roidb, scale_inds, max_phrases, gt_inds):
  """Builds an input blob from the images in the roidb at the specified
  scales.
  """
  num_images = len(roidb)
  assert(num_images == 1)
  processed_ims = []
  im_scales = []
  phrases = []
  phrase_feats = np.zeros((max_phrases, roidb[0]['vecs'].shape[1]), dtype=np.float32)
  for i in range(num_images):
    im = cv2.imread(roidb[i]['image'])
    if roidb[i]['flipped']:
      im = im[:, ::-1, :]
    target_size = cfg.TRAIN.SCALES[scale_inds[i]]
    im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                    cfg.TRAIN.MAX_SIZE)
    im_scales.append(im_scale)
    processed_ims.append(im)
    n_phrase = min(len(roidb[i]['processed_phrases']), max_phrases)
    phrase_feats[:n_phrase, :] = roidb[i]['vecs'][gt_inds]
    phrases.append(n_phrase)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, im_scales, phrase_feats, phrases

