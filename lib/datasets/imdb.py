# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import PIL
import pickle
from collections import Counter
from utils.cython_bbox import bbox_overlaps
import numpy as np
import scipy.sparse
from model.config import cfg


class imdb(object):
  """Image database."""

  def __init__(self, name, word_embedding_dict=None, classes=None):
    self._name = name
    self._word_embedding = word_embedding_dict
    self._num_classes = 0
    if not classes:
      self._classes = []
    else:
      self._classes = classes
    self._image_index = []
    self._obj_proposer = 'gt'
    self._roidb = None
    self._roidb_handler = self.default_roidb
    # Use this dict for storing dataset specific config options
    self.config = {}
    self._processed_phrases = None

  @property
  def name(self):
    return self._name

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def num_phrases(self):
    return len(self._processed_phrases)

  @property
  def phrases(self):
    return self._processed_phrases
    
  @property
  def raw_phrases(self):
    phrases = set()
    for roi in self.roidb:
      phrases.update(roi['phrases'])

    return list(phrases)

  @property
  def image_index(self):
    return self._image_index

  @property
  def roidb_handler(self):
    return self._roidb_handler

  @roidb_handler.setter
  def roidb_handler(self, val):
    self._roidb_handler = val

  def set_proposal_method(self, method):
    method = eval('self.' + method + '_roidb')
    self.roidb_handler = method

  @property
  def roidb(self):
    # A roidb is a list of dictionaries, each with the following keys:
    #   boxes
    #   gt_overlaps
    #   gt_classes
    #   flipped
    if self._roidb is not None:
      return self._roidb
    self._roidb = self.roidb_handler()
    return self._roidb

  @property
  def cache_path(self):
    cache_path = osp.abspath(osp.join(cfg.DATA_DIR, 'cache'))
    if not osp.exists(cache_path):
      os.makedirs(cache_path)
    return cache_path

  @property
  def num_images(self):
    return len(self.image_index)

  def get_word_embedding(self, phrase):
    return self._phrase2vec[phrase]

  def set_roidb_info(self):
    self._phrase2vec = {}
    all_processed_phrases = set()
    for roi in self.roidb:
      roi['vecs'] = []
      roi['processed_phrases'] = []
      for raw_phrase in roi['phrases']:
        if raw_phrase.strip() and self._word_embedding is not None:
          phrase, vec = self._word_embedding[raw_phrase]
        else:
          phrase = 'unk'
          vec = np.zeros(cfg.TEXT_FEAT_DIM, np.float32)

        all_processed_phrases.add(phrase)
        self._phrase2vec[phrase] = vec
        roi['vecs'].append(vec)
        roi['processed_phrases'].append(phrase)

      if len(roi['vecs']) > 0:
        roi['vecs'] = np.vstack(roi['vecs'])

    self._processed_phrases = list(all_processed_phrases)
    self._phrase2token = {}
    for phrase in self._processed_phrases:
      self._phrase2token[phrase] = set(phrase.split())

    self._phrase_to_ind = dict(list(zip(all_processed_phrases, list(range(self.num_phrases)))))

    self._augmented_dictionary = {}
    if cfg.AUGMENTED_POSITIVE_PHRASES:
      fn = osp.join(self.cache_path, self.name + '_corrected.txt')
      with open(fn, 'r') as f:
        for line in f:
          line = line.strip().split(',')
          if len(line) > 1:
            self._augmented_dictionary[line[0]] = line[1:]

    vocab_filename = osp.join(cfg.DATA_DIR, 'cache', self.name.split('_')[0] + '_vocab.pkl')
    if osp.exists(vocab_filename):
      self._train_counts = pickle.load(open(vocab_filename, 'r'))['train_counts']

  def image_path_at(self, i):
    raise NotImplementedError

  def default_roidb(self):
    raise NotImplementedError

  def add_augmented_phrases(self):
    if cfg.AUGMENTED_POSITIVE_PHRASES:
      for roi in self.roidb:
        boxes = []
        vecs = []
        for phrase, box in zip(list(roi['processed_phrases']), roi['boxes']):
          if phrase in self._augmented_dictionary:
            aug_phrase = np.random.choice(self._augmented_dictionary[phrase])
            #for aug_phrase in self._augmented_dictionary[phrase]:
            vecs.append(self._phrase2vec[aug_phrase])
            roi['processed_phrases'].append(aug_phrase)
            boxes.append(box)

        if len(boxes) > 0:
          roi['vecs'] = np.vstack((roi['vecs'], vecs))
          roi['boxes'] = np.vstack((roi['boxes'], boxes))

  def get_ap(self, all_boxes, phrase_start, phrase_end, output_dir=None):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    imdir = '/research/diva2/retrieved_sentences/'
    with open(imdir + 'test.txt', 'r') as f:
      images = []
      for line in f:
        images.append(line.strip())

    scores = np.transpose(np.load(imdir + 'train_scores.npy'))

    with open(imdir + 'train_sentences.txt', 'r') as f:
      sentences = []
      for line in f:
        sentences.append(set(line.strip().split()))

    N = 75
    im2tok = {}
    for im, im_scores in zip(images, scores):
      ind = np.argpartition(im_scores, N)[:N]
      tokens = set()
      for i in ind:
        tokens.update(sentences[i])

      im2tok[im] = tokens

    # For each image get the score and label for the top prediction
    # for every phrase
    gt_scores = [[] for _ in range(self.num_phrases)]
    gt_labels = [[] for _ in range(self.num_phrases)]
    phrase_counts = Counter()
    top1acc = 0.
    total_aug = 0.
    top1acc_aug = 0.
    for index, im_boxes in enumerate(all_boxes):
      tokens = im2tok[self._im_ids[self._image_index[index]]]
      roi = self.roidb[self._image_index[index]]
      phrases_seen = list()
      i = 0
      for gt, phrase in zip(roi['boxes'], roi['processed_phrases']):
        phrase_index = self._phrase_to_ind[phrase]
        gt = gt.reshape((1, 4)).astype(np.float)
        if phrase_index >= phrase_start and phrase_index < phrase_end:
          phrases_seen.append(phrase_index)
          boxes = im_boxes[phrase_index - phrase_start]
          assert(boxes.shape[1] == 5)
          pred = boxes[:, :-1].reshape((-1, 4)).astype(np.float)
          overlaps = bbox_overlaps(pred, gt)
          labels = np.zeros(len(overlaps), np.float32)
          ind = np.where(overlaps >= cfg.TEST.SUCCESS_THRESH)[0]
          if len(ind) > 1:
            ind = min(ind)

          labels[ind] = 1
          top1acc += labels[0]
          gt_labels[phrase_index] += list(labels)
          if len(self._phrase2token[phrase].intersection(tokens)) > 0:
            gt_scores[phrase_index] += list(boxes[:, -1])
          else:
            gt_scores[phrase_index] += list(np.ones(len(labels), np.float32) * -np.inf)

        if phrase in self._augmented_dictionary:
          for aug_phrase in self._augmented_dictionary[phrase]:
            phrase_index = self._phrase_to_ind[aug_phrase]
            if phrase_index >= phrase_start and phrase_index < phrase_end:
              phrases_seen.append(phrase_index)
              boxes = im_boxes[phrase_index - phrase_start]
              
              pred = boxes[:, :-1].reshape((-1, 4)).astype(np.float)
              overlaps = bbox_overlaps(pred, gt)
              labels = np.zeros(len(overlaps), np.float32)
              ind = np.where(overlaps >= cfg.TEST.SUCCESS_THRESH)[0]
              if len(ind) > 1:
                ind = min(ind)

              labels[ind] = 1
              top1acc_aug += labels[0]
              total_aug += 1
              gt_labels[phrase_index] += list(labels)
              if len(self._phrase2token[aug_phrase].intersection(tokens)) > 0:
                gt_scores[phrase_index] += list(boxes[:, -1])
              else:
                gt_scores[phrase_index] += list(np.ones(len(labels), np.float32) * -np.inf)

      phrase_counts.update(phrases_seen)
      phrases_seen = set(phrases_seen)
      for phrase_index, boxes in zip(range(phrase_start, phrase_end), im_boxes):
        if phrase_index not in phrases_seen:
          if len(self._phrase2token[self._processed_phrases[phrase_index]].intersection(tokens)) > 0:
            gt_scores[phrase_index] += list(boxes[:, -1])
          else:
            gt_scores[phrase_index] += list(np.ones(len(labels), np.float32) * -np.inf)

          gt_labels[phrase_index] += list(np.zeros(len(boxes), np.float32))

    # Compute average precision
    ap = np.zeros(self.num_phrases, np.float32)
    for phrase_index in range(phrase_start, phrase_end):
      phrase_labels = gt_labels[phrase_index]
      phrase_scores = gt_scores[phrase_index]
      order = np.argsort(phrase_scores)
      phrase_labels = np.array([phrase_labels[i] for i in order])
      pos_labels = np.where(phrase_labels)[0]
      n_pos = len(pos_labels)
      c = 0
      if n_pos > 0:
        # take into account ground truth phrases which were not
        # correctly localized
        n_missing = phrase_counts[phrase_index] - n_pos
        prec = [(n_pos - i) / float(len(phrase_labels) - index) for i, index in enumerate(pos_labels)]
        rec = [(n_pos - i) / float(n_pos + n_missing) for i, _ in enumerate(pos_labels)]
        c = np.sum([(rec[i] - rec[i+1])*prec[i] for i in range(len(pos_labels)-1)]) + prec[-1]*rec[-1]

      ap[phrase_index] = c

    return ap, phrase_counts, top1acc, total_aug, top1acc_aug

  def evaluate_detections(self, ap, phrase_counts, top1acc, total_aug, top1acc_aug):
    """
    all_boxes is a list of length number-of-classes.
    Each list element is a list of length number-of-images.
    Each of those list elements is either an empty list []
    or a numpy array of detection.

    all_boxes[class][image] = [] or np.array of shape #dets x 5
    """
    # organize mAP by the number of occurrences
    count_thresholds = cfg.TEST.PHRASE_COUNT_THRESHOLDS
    mAP = np.zeros(len(count_thresholds))
    occurrences = np.zeros_like(mAP)
    samples = np.zeros_like(mAP)
    count_index = 0
    for phrase, phrase_index in self._phrase_to_ind.iteritems():
      n_occurrences = phrase_counts[phrase_index]
      if n_occurrences < 1:
        continue

      train_count = 0
      if phrase in self._train_counts:
        train_count = self._train_counts[phrase]

      count_index = min(np.where(train_count <= count_thresholds)[0])
      mAP[count_index] += ap[phrase_index]
      occurrences[count_index] += 1
      samples[count_index] += n_occurrences

    mAP = mAP / occurrences
    thresh_string = '\t'.join([str(thresh) for thresh in count_thresholds])
    print('\nThresholds:  \t' + thresh_string + '\tOverall')

    ap_string = '\t'.join(['%.1f' % round(t * 100, 2) for t in mAP])
    print('AP:            \t' + ap_string + '\t%.1f' % round(np.mean(mAP) * 100, 2))

    occ_string = '\t'.join(['%i' % occ for occ in occurrences])
    print('Per Thresh Cnt:\t' + occ_string + '\t%i' % np.sum(occurrences))

    n_total = np.sum(samples)
    sample_string = '\t'.join(['%i' % item for item in samples])
    print('Instance Cnt:  \t' + sample_string + '\t%i' % n_total)

    acc = round((top1acc/(n_total - total_aug))*100, 2)
    print('Orig Localization Accuracy: %.2f' % acc)
    if cfg.AUGMENTED_POSITIVE_PHRASES:
      acc = round(((top1acc + top1acc_aug)/n_total)*100, 2)
      print('Orig + Augmented Localization Accuracy: %.2f' % acc)

    if cfg.TOP_K_PER_PHRASE > 1:
      n_correct = np.sum([np.sum(item) for item in gt_labels])
      acc = round((n_correct/n_total)*100, 2)
      print('Portion of phrases with good boxes: %.2f\n' % acc)

    return np.mean(mAP)

  def _get_widths(self):
    return [PIL.Image.open(self.image_path_at(i)).size[0]
            for i in range(self.num_images)]

  def append_flipped_images(self):
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'boxes': boxes,
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'gt_classes': self.roidb[i]['gt_classes'],
               'flipped': True}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def evaluate_recall(self, candidate_boxes=None, thresholds=None,
                      area='all', limit=None):
    """Evaluate detection proposal recall metrics.

    Returns:
        results: dictionary of results with keys
            'ar': average recall
            'recalls': vector recalls at each IoU overlap threshold
            'thresholds': vector of IoU overlap thresholds
            'gt_overlaps': vector of all ground-truth overlaps
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {'all': 0, 'small': 1, 'medium': 2, 'large': 3,
             '96-128': 4, '128-256': 5, '256-512': 6, '512-inf': 7}
    area_ranges = [[0 ** 2, 1e5 ** 2],  # all
                   [0 ** 2, 32 ** 2],  # small
                   [32 ** 2, 96 ** 2],  # medium
                   [96 ** 2, 1e5 ** 2],  # large
                   [96 ** 2, 128 ** 2],  # 96-128
                   [128 ** 2, 256 ** 2],  # 128-256
                   [256 ** 2, 512 ** 2],  # 256-512
                   [512 ** 2, 1e5 ** 2],  # 512-inf
                   ]
    assert area in areas, 'unknown area range: {}'.format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = np.zeros(0)
    num_pos = 0
    for i in range(self.num_images):
      # Checking for max_overlaps == 1 avoids including crowd annotations
      # (...pretty hacking :/)
      max_gt_overlaps = self.roidb[i]['gt_overlaps'].toarray().max(axis=1)
      gt_inds = np.where((self.roidb[i]['gt_classes'] > 0) &
                         (max_gt_overlaps == 1))[0]
      gt_boxes = self.roidb[i]['boxes'][gt_inds, :]
      gt_areas = self.roidb[i]['seg_areas'][gt_inds]
      valid_gt_inds = np.where((gt_areas >= area_range[0]) &
                               (gt_areas <= area_range[1]))[0]
      gt_boxes = gt_boxes[valid_gt_inds, :]
      num_pos += len(valid_gt_inds)

      if candidate_boxes is None:
        # If candidate_boxes is not supplied, the default is to use the
        # non-ground-truth boxes from this roidb
        non_gt_inds = np.where(self.roidb[i]['gt_classes'] == 0)[0]
        boxes = self.roidb[i]['boxes'][non_gt_inds, :]
      else:
        boxes = candidate_boxes[i]
      if boxes.shape[0] == 0:
        continue
      if limit is not None and boxes.shape[0] > limit:
        boxes = boxes[:limit, :]

      overlaps = bbox_overlaps(boxes.astype(np.float),
                               gt_boxes.astype(np.float))

      _gt_overlaps = np.zeros((gt_boxes.shape[0]))
      for j in range(gt_boxes.shape[0]):
        # find which proposal box maximally covers each gt box
        argmax_overlaps = overlaps.argmax(axis=0)
        # and get the iou amount of coverage for each gt box
        max_overlaps = overlaps.max(axis=0)
        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ind = max_overlaps.argmax()
        gt_ovr = max_overlaps.max()
        assert (gt_ovr >= 0)
        # find the proposal box that covers the best covered gt box
        box_ind = argmax_overlaps[gt_ind]
        # record the iou coverage of this gt box
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]
        assert (_gt_overlaps[j] == gt_ovr)
        # mark the proposal box and the gt box as used
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1
      # append recorded iou coverage level
      gt_overlaps = np.hstack((gt_overlaps, _gt_overlaps))

    gt_overlaps = np.sort(gt_overlaps)
    if thresholds is None:
      step = 0.05
      thresholds = np.arange(0.5, 0.95 + 1e-5, step)
    recalls = np.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
      recalls[i] = (gt_overlaps >= t).sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {'ar': ar, 'recalls': recalls, 'thresholds': thresholds,
            'gt_overlaps': gt_overlaps}

  def create_roidb_from_box_list(self, box_list, gt_roidb):
    assert len(box_list) == self.num_images, \
      'Number of boxes must match number of ground-truth images'
    roidb = []
    for i in range(self.num_images):
      boxes = box_list[i]
      num_boxes = boxes.shape[0]
      overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

      if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
        gt_boxes = gt_roidb[i]['boxes']
        gt_classes = gt_roidb[i]['gt_classes']
        gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                    gt_boxes.astype(np.float))
        argmaxes = gt_overlaps.argmax(axis=1)
        maxes = gt_overlaps.max(axis=1)
        I = np.where(maxes > 0)[0]
        overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

      overlaps = scipy.sparse.csr_matrix(overlaps)
      roidb.append({
        'boxes': boxes,
        'gt_classes': np.zeros((num_boxes,), dtype=np.int32),
        'gt_overlaps': overlaps,
        'flipped': False,
        'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
      })
    return roidb

  @staticmethod
  def merge_roidbs(a, b):
    assert len(a) == len(b)
    for i in range(len(a)):
      a[i]['boxes'] = np.vstack((a[i]['boxes'], b[i]['boxes']))
      a[i]['gt_classes'] = np.hstack((a[i]['gt_classes'],
                                      b[i]['gt_classes']))
      a[i]['gt_overlaps'] = scipy.sparse.vstack([a[i]['gt_overlaps'],
                                                 b[i]['gt_overlaps']])
      a[i]['seg_areas'] = np.hstack((a[i]['seg_areas'],
                                     b[i]['seg_areas']))
    return a

  def competition_mode(self, on):
    """Turn competition mode on or off."""
    pass
