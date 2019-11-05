from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import os
import os.path as osp
import numpy as np

from cite.data_loader import load_word_embeddings

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0001

# Weight for L2 regularization when training CCA-initialized layers
__C.TRAIN.CCA_L2_REG = 5e-5

# Margin used to compute triplet loss for embedding classifier
__C.TRAIN.EMBED_MARGIN = 0.05

# Number of violuations per-phrase used to compute triplet loss for embedding classifier
__C.TRAIN.EMBED_TOPK_VIOLATIONS = 5

# Weight for (phrase, phrase, region) triplets
__C.TRAIN.EMBED_PHRASE_PHRASE_REGION_LOSS = 1.0

# Weight for (phrase, region, region) triplets
__C.TRAIN.EMBED_PHRASE_REGION_REGION_LOSS = 4.0

# Weight for the region neighborhood constraint
__C.TRAIN.EMBED_REGION_LOSS = 0.1

# Weight for the phrase neighborhood constraint
__C.TRAIN.EMBED_PHRASE_LOSS = 0.1

# All phrases to be a single class when creating triplets for the bidirectional loss
__C.TRAIN.EMBED_SINGLE_CATEGORY = True

# Output embedding dimensions for CITE classifier
__C.TRAIN.CITE_FINAL_EMBED = 256

# Number of negatives for every positive used to train the CITE classifier
__C.TRAIN.CITE_NEG_TO_POS_RATIO = 2

# Number of concepts to train for CITE classifier
__C.TRAIN.CITE_NUM_CONCEPTS = 4

# Concept weight branch L1 loss weight for CITE classifier
__C.TRAIN.CITE_CONCEPT_LOSS_WEIGHT = 5e-5

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# Whether to double the learning rate for bias
__C.TRAIN.DOUBLE_BIAS = True

# Whether to initialize the weights with truncated normal distribution 
__C.TRAIN.TRUNCATED = False

# Whether to have weight decay on bias as well
__C.TRAIN.BIAS_DECAY = False

# Whether to add ground truth boxes to the pool when sampling regions
__C.TRAIN.USE_GT = False

# Whether to use aspect-ratio grouping of training images, introduced merely for saving
# GPU memory
__C.TRAIN.ASPECT_GROUPING = False

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = 3

# The time interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 180

# Scale to use during training (can list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = False

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 5000

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True

__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)

__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'gt'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Maximum number of phrases paired with each image during training
__C.TRAIN.MAX_PHRASES = 5

#
# Testing options
#
__C.TEST = edict()

# Scale to use during testing (can NOT list multiple scales)
# The scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'gt'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

## Threshold used to determine if a prediction is correct or not
__C.TEST.SUCCESS_THRESH = 0.5

# Minimum phrase occurrence thresholds used for detection evaluation
# Should be in decreasing order
__C.TEST.PHRASE_COUNT_THRESHOLDS = np.array([0, 100, np.inf])

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
# __C.TEST.RPN_MIN_SIZE = 16

# Testing mode, default to be 'nms', 'top' is slower but better
# See report for details
__C.TEST.MODE = 'nms'

# Only useful when TEST.MODE is 'top', specifies the number of top proposals to select
__C.TEST.RPN_TOP_N = 5000

# Maximum number of phrases paired with each image at test time
#__C.TEST.MAX_PHRASES = 400
__C.TEST.MAX_PHRASES = 650

#
# ResNet options
#

__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1

#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default pooling mode, only 'crop' is available
__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Default region classifier/loss type (entropy, cite, embed, classifier)
__C.REGION_CLASSIFIER = 'cite'

# The prefix used to identify the word embedding vocabulary
__C.WORD_EMBEDDING_TYPE = 'hglmm'

# Dimensions of the input text features (i.e. what size the embedding
# type features are)
__C.WORD_EMBEDDING_DIMENSIONS = {'hglmm' : 6000, 'fasttext' : 300}

# Use CCA to initialize the classification layers, only valid for 
# "embed" classifier
__C.CCA_INIT = True

# Test a CCA model, only works when CCA_INIT is true
__C.TEST_CCA = False

# Use CCA to initialize the classification layers, only valid for 
# "embed" classifier
__C.EMBED_LAYERS = [-1]

# Concat 5-D box features consisting of normalized [x1, y1, x2, y2, area]
# to inputs of the classifier layers
__C.BBOX_FEATS = True

# Indicates whether to use just the ground truth annotated phrases or
# if augmented set from tokenization and WordNet should be used
__C.AUGMENTED_POSITIVE_PHRASES = True

# Output dimensions of the desired word embedding features.  If
# TEXT_FEAT_DIM != WORD_EMBEDDING_DIMENSIONS, a fully connected layer
# is used to project the initial embeddings to the same dimensions
__C.TEXT_FEAT_DIM = 6000

# Number of predictions made per-phrase
__C.TOP_K_PER_PHRASE = 1

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512

def get_output_vocab(imdb_name):
  dataname = imdb_name.split('_')[0]
  vocab_name = dataname + '_vocab.pkl'
  cached_vocab = osp.join(__C.DATA_DIR, 'cache', vocab_name)
  vocab = pickle.load(open(cached_vocab, 'rb'))

  # a map from the raw phrases in the processed phrases
  corrected_phrases = vocab['corrected']

  # The following assumes each corrected phrase from above has an embedding in
  # the embedding text file.  Each row's entries are comma separated, where the
  # first entry is the phrase, and the next K are the embedding vector's values
  cached_embedding = '/research/diva2/word_vectors/fivetask_hglmm_pca6000.txt'
  #cached_embedding = 'data/flickr_fasttext_embedding.txt'
  embedding_dictionary = {}
  word_embedding_dims = __C.WORD_EMBEDDING_DIMENSIONS[__C.WORD_EMBEDDING_TYPE]
  tok2idx, vecs = load_word_embeddings(cached_embedding, word_embedding_dims)

  max_tokens = 0
  final_embedding_dictionary = {}
  for raw, label in corrected_phrases.iteritems():
    indices = [tok2idx[w] for w in label.split() if w in tok2idx]
    max_tokens = max(max_tokens, len(indices))
    final_embedding_dictionary[raw] = (label, indices)

  final_embedding_dictionary['vecs'] = vecs
  final_embedding_dictionary['max_tokens'] = max_tokens
  for raw in corrected_phrases:
    label, indices = final_embedding_dictionary[raw]
    vec = np.zeros(max_tokens, np.int32)
    vec[:len(indices)] = indices
    final_embedding_dictionary[raw] = (label, vec)

  #print('embedding %s loaded' % embedding_filename)
  return final_embedding_dictionary

def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir

def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():
    # a must specify keys that are in b
    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))

    # the types must match, too
    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))

    # recursively merge dicts
    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:
      # handle the case when v is a string literal
      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value
