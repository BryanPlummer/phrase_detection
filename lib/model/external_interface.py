from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from model.config import cfg

import os
import numpy as np
import tensorflow as tf

from cite.model import encode_regions, encode_phrases, get_max_phrase_scores

class CiteArgs():
    def __init__(self, cca_cache_dir):
        self.dim_embed = cfg.TRAIN.CITE_FINAL_EMBED
        self.batch_size = 1
        self.cca_parameters = ''
        if cca_cache_dir is not None:
            self.cca_parameters = os.path.join(cca_cache_dir, 'cca_parameters_512.pkl')

        self.cca_weight_reg = cfg.TRAIN.CCA_L2_REG
        self.two_branch = cfg.REGION_CLASSIFIER == 'cite_embed'
        self.language_model = 'avg'

        # code computing scores assumes a single image at a time is processed
        assert(self.batch_size == 1)
        self.num_embeddings = cfg.TRAIN.CITE_NUM_CONCEPTS

def get_projected_regions(args, model_filename, features):
    region_feature_dim = features[0].shape[1]
    region_plh = tf.placeholder(tf.float32, shape=[args.batch_size, None, region_feature_dim])
    train_phase_plh = tf.placeholder(tf.bool, name='train_phase')
    num_boxes_plh = tf.placeholder(tf.int32)
    # first, lets make the features smaller
    regions = encode_regions(args, region_plh, train_phase_plh, num_boxes_plh, region_feature_dim)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
    region_embed = []
    for im_feats in tqdm(features, desc='computing projected regions', total=len(features)):
        feed_dict = {region_plh : np.expand_dims(im_feats, 0),
                     num_boxes_plh : len(im_feats),
                     train_phase_plh : False}
        region_embed.append(sess.run(regions, feed_dict=feed_dict))

    sess.close()
    return region_embed

def get_projected_phrases(args, imdb, model_filename, vecs, max_length):
    phrase_plh = tf.placeholder(tf.int32, shape=[args.batch_size, None, max_length])
    train_phase_plh = tf.placeholder(tf.bool, name='train_phase')
    num_phrases_plh = tf.placeholder(tf.int32)
    phrase_denom_plh = tf.placeholder(tf.float32)
    phrase_encoding = encode_phrases(args, phrase_plh, train_phase_plh, num_phrases_plh, max_length, phrase_denom_plh, vecs)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
    phrases = imdb.phrases
    n_batches = int(np.ceil(len(phrases) / float(cfg.TEST.MAX_PHRASES)))
    all_phrases, all_concepts = [], []
    eps = 1e-10
    for batch_id in tqdm(range(n_batches), desc='computing projected phrases', total=n_batches):
        phrase_start = batch_id*cfg.TEST.MAX_PHRASES
        phrase_end = min((batch_id+1)*cfg.TEST.MAX_PHRASES, len(phrases))
        phrase = np.vstack([imdb.get_word_embedding(p) for p in phrases[phrase_start:phrase_end]])
        feed_dict = {phrase_plh : np.expand_dims(phrase, 0),
                     num_phrases_plh : len(phrase),
                     phrase_denom_plh : len(phrase) + eps,
                     train_phase_plh : False}

        phrase_embed, concepts, _ = sess.run(phrase_encoding, feed_dict=feed_dict)
        all_phrases.append(phrase_embed.squeeze())
        all_concepts.append(concepts.squeeze())

    all_phrases = np.vstack(all_phrases)
    all_concepts = np.vstack(all_concepts)
    sess.close()
    return all_phrases, all_concepts

def get_cite_scores(imdb, model_filename, rois, features, im_shapes, cca_cache_dir, vecs, max_length):
    args = CiteArgs(cca_cache_dir)
    features = get_projected_regions(args, model_filename, features)
    phrases, concepts = get_projected_phrases(args, imdb, model_filename, vecs, max_length)
    region_embed_plh = tf.placeholder(tf.float32, shape=[args.batch_size, None, features[0].shape[2]])
    phrase_embed_plh = tf.placeholder(tf.float32, shape=[None, phrases.shape[1]])
    phrase_concepts_plh = tf.placeholder(tf.float32, shape=[None, concepts.shape[1]])
    num_phrases_plh = tf.placeholder(tf.int32)
    num_boxes_plh = tf.placeholder(tf.int32)
    train_phase_plh = tf.placeholder(tf.bool, name='train_phase')
    model_scores = get_max_phrase_scores(args, phrase_embed_plh, phrase_concepts_plh, num_phrases_plh, region_embed_plh, train_phase_plh, num_boxes_plh, phrases.shape[1])
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
    phrase_scores = np.zeros((len(features), len(phrases), 2), np.float32)
    n_phrase_batches = int(np.ceil(len(phrases) / float(cfg.TEST.MAX_PHRASES)))
    for phrase_batch_id in range(n_phrase_batches):
        phrase_start = phrase_batch_id*cfg.TEST.MAX_PHRASES
        phrase_end = min((phrase_batch_id+1)*cfg.TEST.MAX_PHRASES, len(phrases))
        phrase_embed = phrases[phrase_start:phrase_end]
        phrase_concepts = concepts[phrase_start:phrase_end]
        feed_dict = {phrase_embed_plh : phrase_embed,
                     phrase_concepts_plh : phrase_concepts,
                     num_phrases_plh : len(phrase_embed),
                     train_phase_plh : False}

        for i, im_features in tqdm(enumerate(features), 
                                   desc='scoring phrase batch [%i/%i]' % 
                                         (phrase_batch_id + 1, n_phrase_batches),
                                   total=len(features)):
            feed_dict[region_embed_plh] = im_features
            feed_dict[num_boxes_plh] = im_features.shape[1]
            scores, indices = sess.run(model_scores, feed_dict=feed_dict)
            phrase_scores[i, phrase_start:phrase_end, 0] = scores
            phrase_scores[i, phrase_start:phrase_end, 1] = indices

    sess.close()
    return phrase_scores

        
