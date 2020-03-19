# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer based on code from Ross Girshick,
# Zheqi he, and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_vocab
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys, os

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--cca_iters', dest='cca_iters',
                      help='number of iterations before CCA initilization',
                      default=360000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names, vocab):
  """
  Combine multiple roidbs
  """

  def get_roidb(imdb_name, vocab):
    imdb = get_imdb(imdb_name, vocab)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s, vocab) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1], vocab)
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names, vocab)
  return imdb, roidb


if __name__ == '__main__':
  args = parse_args()
  
  print('Called with args:')
  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  np.random.seed(cfg.RNG_SEED)

  vocab = get_output_vocab(args.imdb_name)
  # train set
  imdb, roidb = combined_roidb(args.imdb_name, vocab)
  imdb.add_augmented_phrases()

  print('{:d} roidb entries'.format(len(roidb)))

  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  _, valroidb = combined_roidb(args.imdbval_name, vocab)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip
  cca_cache_dir = None
  if cfg.CCA_INIT:
    cca_cache_dir = os.path.join(output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_' + str(args.cca_iters))

  # load network
  if args.net == 'vgg16':
    net = vgg16(cca_cache_dir, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res50':
    net = resnetv1(num_layers=50, output_dir=cca_cache_dir, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res101':
    net = resnetv1(num_layers=101, output_dir=cca_cache_dir, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res152':
    net = resnetv1(num_layers=152, output_dir=cca_cache_dir, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  else:
    raise NotImplementedError
    
  train_net(net, imdb, roidb, valroidb, output_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
