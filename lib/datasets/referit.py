# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('agg')

from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from model.config import cfg, get_output_vocab
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import json
import uuid
import h5py
import string

class referit(imdb):
  def __init__(self, word_embedding_dict, image_set, data=None):
    imdb.__init__(self, 'referit_' + image_set, word_embedding_dict)
    self._data = data

    # name, paths
    self._image_set = image_set
    self._data_path = osp.join(cfg.DATA_DIR, 'referit')
    self._classes = tuple(['__background__', '__phrase__'])
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

    self._image_index = self._load_image_set_index()

    # Default to roidb handler
    self.set_proposal_method('gt')
    self.set_roidb_info()

    self._image_index = self._load_image_set_index()

  def _load_image_set_index(self):
    """
    Load image ids.
    """
    ref_ids = self._data.getRefIds(split=self._image_set)
    self._im_ids = list(set(self._data.getImgIds(ref_ids)))
    return range(len(self._im_ids))

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """    
    im_id = self._im_ids[self._image_index[i]]
    im_fn = self._data.loadImgs(im_id)[0]['file_name']
    return os.path.join(self._data_path, 'saiapr_tc-12', im_fn)

  def gt_roidb(self):
    """
    Return the database of ground-truth regions of interest.
    This function loads/saves from/to a cache file to speed up future calls.
    """
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    if osp.exists(cache_file):
      with open(cache_file, 'rb') as fid:
        roidb = pickle.load(fid)

      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      return roidb

    image_to_ind = dict(list(zip(self._im_ids, list(range(self.num_images)))))
    gt_roidb = [self._load_referit_annotation(image_to_ind, index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_referit_annotation(self, image_to_ind, image_index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_id = self._im_ids[self._image_index[image_index]]
    refs = self._data.imgToRefs[im_id]
    ref_ids = [ref['ref_id'] for ref in refs]
    gt_phrases = []
    gt_boxes = []
    for ref_id, ref in zip(ref_ids, refs):
      box = self._data.getRefBox(ref_id)
      box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
      for sent_annos in ref['sentences']:
        gt_phrases.append(sent_annos['raw'].encode('ascii','ignore').lower())
        gt_boxes.append(box)

    if len(gt_boxes) > 0:
      gt_boxes = np.vstack(gt_boxes)

    return {'phrases': gt_phrases,
            'boxes': gt_boxes,
            'flipped': False}

