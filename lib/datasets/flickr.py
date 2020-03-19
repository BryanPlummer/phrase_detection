# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from flickr30k_entities_utils import get_sentence_data, get_annotations

class flickr(imdb):
  def __init__(self, word_embedding_dict, image_set):
    imdb.__init__(self, 'flickr_' + image_set, word_embedding_dict)
    # name, paths
    self._image_set = image_set
    self._data_path = osp.join('data', 'flickr')
    self._classes = tuple(['__background__', '__phrase__'])
    self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))

    self._image_index = self._load_image_set_index()

    # Default to roidb handler
    self.set_proposal_method('gt')
    self.set_roidb_info()
    if cfg.TEST.SENTENCE_FILTERING and image_set == 'test':
      rsent_dir = osp.join(self._data_path, 'retrieved_sentences')
      sentence_order = np.loadtxt(osp.join(rsent_dir, 'sentence_order.gz'), np.int32)
      with open(osp.join(rsent_dir, 'train_sentences.txt'), 'r') as f:
        sentences = [set(line.strip().split()) for line in f]
        
      self._phrases_per_image = {}
      for im, order in zip(self._im_ids, sentence_order):
        tokens = set()
        for i in order[:cfg.TEST.SENTENCE_FILTERING]:
          tokens.update(sentences[i])

        self._phrases_per_image[im] = tokens


  def _load_image_set_index(self):
    """
    Load image ids.
    """
    with open(osp.join(self._data_path, self._image_set + '.txt'), 'r') as f:
      self._im_ids = [im_id.strip() for im_id in f.readlines()]

    return range(len(self._im_ids))

  def image_path_at(self, i):
    """
    Return the absolute path to image i in the image sequence.
    """    
    im_id = self._im_ids[self._image_index[i]]
    return os.path.join(self._data_path, 'images', im_id + '.jpg')

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

    gt_roidb = [self._load_flickr_annotation(index)
                for index in self._image_index]

    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    return gt_roidb

  def _load_flickr_annotation(self, image_index):
    """
    Loads COCO bounding-box instance annotations. Crowd instances are
    handled by marking their overlaps (with all categories) to -1. This
    overlap value means that crowd "instances" are excluded from training.
    """
    im_id = self._im_ids[image_index]
    sentence_data = get_sentence_data(osp.join(self._data_path, 'Sentences', im_id + '.txt'))
    annotations = get_annotations(osp.join(self._data_path, 'Annotations', im_id + '.xml'))
    phrase_class_index = self._class_to_ind['__phrase__']
    gt_boxes = []
    gt_phrases = []
    words = []
    for sentence in sentence_data:
      for phrase_info in sentence['phrases']:
        phrase_id = phrase_info['phrase_id']
        if phrase_id in annotations['boxes']:
          phrase = phrase_info['phrase'].lower()
          gt_phrases.append(phrase)
          boxes = np.array(annotations['boxes'][phrase_id])
          gt_box = [min(boxes[:, 0]), min(boxes[:, 1]), max(boxes[:, 2]), max(boxes[:, 3])]
          gt_boxes.append(np.array(gt_box, dtype=np.float32))

    if len(gt_boxes) > 0:
      gt_boxes = np.vstack(gt_boxes)

    return {'phrases': gt_phrases,
            'boxes': gt_boxes,
            'flipped': False}

