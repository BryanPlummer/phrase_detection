# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
from model.config import cfg

import os
import numpy as np
import tensorflow as tf

from cite.model import CITE

class CiteArgs():
    def __init__(self, cca_cache_dir):
        self.dim_embed = cfg.TRAIN.CITE_FINAL_EMBED
        self.batch_size = 1
        self.cca_parameters = ''
        if cca_cache_dir is not None:
            self.cca_parameters = os.path.join(cca_cache_dir, 'cca_parameters.pkl')

        self.cca_weight_reg = cfg.TRAIN.CCA_L2_REG
        self.two_branch = cfg.REGION_CLASSIFIER == 'cite_embed'
        self.language_model = 'avg'
        self.region_norm_axis = 2
        self.embedding_ft = False

        # code computing scores assumes a single image at a time is processed
        assert(self.batch_size == 1)
        self.num_embeddings = cfg.TRAIN.CITE_NUM_CONCEPTS

def get_projected_regions(args, model_filename, vecs, features):
    region_feature_dim = features[0].shape[1]
    model = CITE(args, vecs, region_feature_dim=region_feature_dim)

    # first, lets make the features smaller
    regions = model.encode_regions()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
    region_embed = []
    plh = model.get_region_placeholders()
    for im_feats in tqdm(features, desc='computing projected regions', total=len(features)):
        feed_dict = {plh['regions'] : np.expand_dims(im_feats, 0),
                     plh['boxes_per_image'] : len(im_feats),
                     plh['train_phase'] : False}
        region_embed.append(sess.run(regions, feed_dict=feed_dict))

    sess.close()
    return region_embed

def get_projected_phrases(args, imdb, model_filename, vecs, max_length):
    model = CITE(args, vecs, max_length)
    phrase_encoding = model.encode_phrases()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_filename)
    phrases = imdb.phrases
    n_batches = int(np.ceil(len(phrases) / float(cfg.TEST.MAX_PHRASES)))
    all_phrases, all_concepts = [], []
    eps = 1e-10
    plh = model.get_phrase_placeholders()
    for batch_id in tqdm(range(n_batches), desc='computing projected phrases', total=n_batches):
        phrase_start = batch_id*cfg.TEST.MAX_PHRASES
        phrase_end = min((batch_id+1)*cfg.TEST.MAX_PHRASES, len(phrases))
        phrase = np.vstack([imdb.get_word_embedding(p) for p in phrases[phrase_start:phrase_end]])
        feed_dict = {plh['phrases'] : np.expand_dims(phrase, 0),
                     plh['phrases_per_image'] : len(phrase),
                     plh['phrase_count'] : len(phrase) + eps,
                     plh['train_phase'] : False}

        phrase_embed, concepts, _, _ = sess.run(phrase_encoding, feed_dict=feed_dict)
        all_phrases.append(phrase_embed.squeeze())
        all_concepts.append(concepts.squeeze())

    all_phrases = np.vstack(all_phrases)
    all_concepts = np.vstack(all_concepts)
    sess.close()
    return all_phrases, all_concepts

def get_cite_scores(imdb, model_filename, rois, features, im_shapes, cca_cache_dir, vecs, max_length):
    args = CiteArgs(cca_cache_dir)
    features = get_projected_regions(args, model_filename, vecs, features)
    phrases, concepts = get_projected_phrases(args, imdb, model_filename, vecs, max_length)

    model = CITE(args)
    plh = model.get_placeholders()
    embed_dim = phrases.shape[1]
    region_embed_plh = tf.placeholder(tf.float32, shape=[args.batch_size, None, embed_dim])
    phrase_embed_plh = tf.placeholder(tf.float32, shape=[None, embed_dim])
    phrase_concepts_plh = tf.placeholder(tf.float32, shape=[None, concepts.shape[1]])
    model_scores = model.get_max_phrase_scores(phrase_embed_plh, phrase_concepts_plh, region_embed_plh, embed_dim)

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
                     plh['phrases_per_image'] : len(phrase_embed),
                     plh['train_phase'] : False}

        for i, im_features in tqdm(enumerate(features), 
                                   desc='scoring phrase batch [%i/%i]' % 
                                         (phrase_batch_id + 1, n_phrase_batches),
                                   total=len(features)):
            feed_dict[region_embed_plh] = im_features
            feed_dict[plh['boxes_per_image']] = im_features.shape[1]
            scores, indices = sess.run(model_scores, feed_dict=feed_dict)
            phrase_scores[i, phrase_start:phrase_end, 0] = scores
            phrase_scores[i, phrase_start:phrase_end, 1] = indices

    sess.close()
    return phrase_scores

        
