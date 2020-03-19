# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp

from model.config import cfg

__sets = {}
from datasets.flickr import flickr
from datasets.referit import referit

from refer import REFER

for split in ['train', 'test', 'val']:
  name = 'flickr_{}'.format(split)
  __sets[name] = (lambda vocab, split=split: flickr(vocab, split))

refer = REFER(cfg.DATA_DIR, dataset='refclef',  splitBy='berkeley')
for split in ['train', 'test', 'val']:
  name = 'referit_{}'.format(split)
  __sets[name] = (lambda vocab, split=split: referit(vocab, split, data=refer))

def get_imdb(name, vocab = None):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name](vocab)

def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
