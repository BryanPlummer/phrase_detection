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
from utils.timer import Timer
from utils.blob import im_list_to_blob

from model.train_val import filter_roidb
from model.feature_extractor import im_detect
from model.config import cfg, get_output_dir
from model.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps

def linear_cca(H1, H2, outdim_size):
    """
    An implementation of linear CCA taken from:
      https://github.com/VahidooX/DeepCCA/blob/master/linear_cca.py

    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o1 = H1.shape[1]
    o2 = H2.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o2)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]

    return A, B, mean1, mean2, D

def train_cca_model(sess, net, imdb, weights_filename):
  np.random.seed(cfg.RNG_SEED)
  """Test a Fast R-CNN network on an image database."""
  output_dir = get_output_dir(imdb, weights_filename)
  feat_cache_dir = os.path.join(output_dir, 'cca_features')
  if not os.path.exists(feat_cache_dir):
    os.makedirs(feat_cache_dir)

  roidb = filter_roidb(imdb.roidb)
  for i in tqdm(range(len(roidb)), desc='computing cca features', total=len(roidb)):
    imname = roidb[i]['image'].split('.')[0].split(os.path.sep)[-1]
    det_file = os.path.join(feat_cache_dir,  imname + '.pkl')
    if os.path.exists(det_file):
      continue

    im = cv2.imread(roidb[i]['image'])
    phrases = roidb[i]['vecs']
    im_boxes, features, phrases = im_detect(sess, net, im, phrases)
    good_phrases = []
    good_regions = []
    for phrase, gt, rois in zip(phrases, roidb[i]['boxes'], im_boxes):
      rois = rois.astype(np.float)
      gt = gt.reshape((1, 4)).astype(np.float)
      overlaps = [bbox_overlaps(box[:-1].reshape((1, 4)), gt) for box in rois]
      overlaps = np.vstack(overlaps).squeeze()
      index = np.argmax(overlaps)
      if overlaps[index] >= 0.6:
        good_phrases.append(phrase)
        good_regions.append(features[index])

    if len(good_regions) > 0:
      good_regions = np.vstack(good_regions).astype(np.float32)
      good_phrases = np.vstack(good_phrases).astype(np.float32)
      cca_features = {'vis' : good_regions, 'lang' : good_phrases}    
      with open(det_file, 'wb') as f:
        pickle.dump(cca_features, f, pickle.HIGHEST_PROTOCOL)

  all_regions = []
  all_phrases = []
  outdim = 2048
  if cfg.BBOX_FEATS:
    # box feats are 5-D
    outdim += 5

  for i in tqdm(range(len(roidb)), desc='loading cached features', total=len(roidb)):
    imname = roidb[i]['image'].split('.')[0].split(os.path.sep)[-1]
    det_file = os.path.join(feat_cache_dir, imname + '.pkl')
    if os.path.exists(det_file):
      with open(det_file, 'rb') as f:
        cca_features = pickle.load(f)

      n_items = int(float(len(cca_features['vis'])))# // 3.)
      all_regions.append(cca_features['vis'][:n_items, :outdim])
      all_phrases.append(cca_features['lang'][:n_items])

  all_regions = np.vstack(all_regions).astype(np.float32)
  all_phrases = np.vstack(all_phrases).astype(np.float32)

  assert(len(all_regions) == len(all_phrases))
  all_layers = []
  for layer_id, outdim_size in enumerate(cfg.EMBED_LAYERS):
    max_size = min(all_phrases.shape[1], all_regions.shape[1])
    if outdim_size < 1:
      outdim_size = max_size
    else:
      outdim_size = min(outdim_size, max_size)

    print('Estimating CCA parameters for {:d} samples'.format(len(all_regions)))
    Wx, Wy, mX, mY, scaling = linear_cca(all_regions, all_phrases, outdim_size)

    cca_parameters = {}
    cca_parameters['vis_proj'] = Wx
    cca_parameters['lang_proj'] = Wy
    cca_parameters['vis_mean'] = mX
    cca_parameters['lang_mean'] = mY
    cca_parameters['scaling'] = scaling
    all_layers.append(cca_parameters)

    # get the next layer's inputs if this isn't the last layer
    if (layer_id + 1) < len(cfg.EMBED_LAYERS):
      all_regions = np.dot(all_regions - mX, Wx) * scaling
      all_regions[all_regions < 0] = 0
      all_phrases = np.dot(all_phrases - mY, Wy) * scaling
      all_phrases[all_phrases < 0] = 0

  print('done, saving...')
  det_file = os.path.join(output_dir, 'cca_parameters.pkl')
  with open(det_file, 'wb') as f:
    pickle.dump(all_layers, f, pickle.HIGHEST_PROTOCOL)
  print('completed CCA parameter estimation')
