# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test_largevocab import get_features, test_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_vocab
from model.external_interface import get_cite_scores
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--cca_iters', dest='cca_iters',
                      help='number of iterations before CCA initilization',
                      default=360000, type=int)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
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

def restore_model(args, sess, variables_to_restore = None):
  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    if variables_to_restore is None:
      saver = tf.train.Saver()
    else:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver(variables_to_restore)

    saver.restore(sess, args.model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')

if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename
  vocab = get_output_vocab(args.imdb_name)
  imdb = get_imdb(args.imdb_name, vocab)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # used to load CCA parameters
  output_dir = get_output_dir(imdb, args.tag)
  output_dir = output_dir.replace('test', 'train').replace('val', 'train')
  cca_cache_dir = None
  if cfg.CCA_INIT:
    cca_cache_dir = os.path.join(output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_' + str(args.cca_iters))

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16(cca_cache_dir, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res50':
    net = resnetv1(cca_cache_dir, num_layers=50, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res101':
    net = resnetv1(cca_cache_dir, num_layers=101, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  elif args.net == 'res152':
    net = resnetv1(cca_cache_dir, num_layers=152, vecs=vocab['vecs'], max_tokens=vocab['max_tokens'])
  else:
    raise NotImplementedError

  # load model
  net.create_feat_extractor("TEST", imdb.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

  restore_model(args, sess)
  rois, features, im_shapes = get_features(sess, net, imdb)
  sess.close()
  #rois, im_shapes, features = None, None, None
  phrase_scores = None
  if cfg.REGION_CLASSIFIER in ['cite']:
    tf.reset_default_graph()
    model_filename = 'external/cite/runs/%s_cca/model_best' % args.imdb_name.split('_')[0].lower()

    phrase_scores = get_cite_scores(imdb, model_filename, rois, features, im_shapes, cca_cache_dir, vocab['vecs'], vocab['max_tokens'])
    tf.reset_default_graph()
  
  sess = tf.Session(config=tfconfig)
  net.create_phrase_extractor("TEST", imdb.num_classes, tag='default')

  variables_to_restore = None
  if cfg.CCA_INIT and cfg.TEST_CCA:
    # don't try to restore variables that deal with the classifier since
    # we are testing just CCA here
    variables = tf.contrib.slim.get_variables_to_restore()
    #variables_to_restore = [v for v in variables if v.name != 'word_embeddings:0' and v.name != 'resnet_v1_101/word_embeddings' and v.name.split('/')[1].split('_')[0] != 'cls']
    variables_to_restore = [v for v in variables if v.name == 'word_embeddings:0' or v.name.split('/')[1].split('_')[0] != 'cls']
    
  restore_model(args, sess, variables_to_restore)
  test_net(sess, net, imdb, rois, features, im_shapes, filename, phrase_scores=phrase_scores)
  sess.close()




