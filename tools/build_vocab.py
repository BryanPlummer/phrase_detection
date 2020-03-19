# --------------------------------------------------------
# Tensorflow Phrase Detection
# Licensed under The MIT License [see LICENSE for details]
# Written by Bryan Plummer
# Includes code from http://norvig.com/spell-correct.html
# --------------------------------------------------------
import _init_paths

import pickle
import argparse
import os
import re
import string
import numpy as np
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn

from collections import Counter

import roi_data_layer.roidb as rdl_roidb
from datasets.factory import get_imdb

"""
Parse input arguments
"""
parser = argparse.ArgumentParser(description='Build a vocabularly')
parser.add_argument('--dataset', dest='dataset',
                    help='dataset to build a vocabularly for',
                    default='flickr', type=str)
parser.add_argument('--embedding', dest='embedding',
                    help='type of word embedding to use (hglmm or fasttext)',
                    default='hglmm', type=str)
parser.add_argument('--remove_stopwords', dest='stopwords', action='store_true',
                    help='do stopword removal when true',
                    default=False)
args = parser.parse_args()

with open(os.path.join('data','word2vec_vocab.txt'), 'r') as f:
    embedding_tokens = [token.strip().lower() for token in f.readlines()]

# Begin spell-checking code from http://norvig.com/spell-correct.html
def words(text): return re.findall(r'\w+', text.lower())
WORDS = Counter(words(' '.join(embedding_tokens)))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# End spell-checking code

# get all the phrases in the dataset 
print('loading dataset phrases')

# code later assumes last split was train so it can use its imdb
splits = ['test', 'val', 'train']
phrases = set()
phrases_per_split = {}
for split in splits:
    imdb_name = args.dataset + '_' + split
    imdb = get_imdb(imdb_name)
    split_phrases = imdb.raw_phrases
    phrases_per_split[split] = split_phrases
    phrases.update(split_phrases)

# Use a stopword dictionary that has been curated for visual grounding
with open(os.path.join('data', 'stopwords.txt'), 'r') as f:
    stopwords = set([word.strip() for word in f.readlines() if word])

# Match each word for each phrase back to their vectors so it uses
# the same pre-processing as our vocabularly
print('mapping phrases to vocab entries')
correction_dictionary = {}
rawphrase2testphrase = {}
for orig_phrase in phrases:
    phrase = orig_phrase.lower()
    corrected_tokens = []
    for raw_token in nltk.tokenize.word_tokenize(phrase):
        if raw_token in correction_dictionary:
            token = correction_dictionary[raw_token]
        else:
            # only fix tokens that are long enough
            if len(raw_token) > 2:
                token = correction(raw_token)
            else:
                token = raw_token

            correction_dictionary[raw_token] = token

        if token:
            corrected_tokens.append(token)

    testphrase = ' '.join(corrected_tokens).translate(None, string.punctuation).strip()
    if args.stopwords:
        # remove stopwords
        filtered_tokens = [token for token in corrected_tokens if token not in stopwords]
        filtered_phrase = ' '.join(filtered_tokens).translate(None, string.punctuation).strip()
        if filtered_phrase:
            testphrase = filtered_phrase
        
    if not testphrase:
        testphrase = 'unk'

    rawphrase2testphrase[orig_phrase] = ' '.join(testphrase.split())

assert(splits[-1] == 'train')
train_counts = Counter()
for roi in imdb.roidb:
    phrases = [rawphrase2testphrase[p] for p in roi['phrases']]
    train_counts.update(phrases)

output_vocab = {'correction_dictionary' : correction_dictionary,
                'corrected' : rawphrase2testphrase,
                'train_counts' : train_counts}

for split in splits:
    corrected_phrases = set([rawphrase2testphrase[p] for p in phrases_per_split[split] if p])
    output_vocab[split + '_corrected'] = list(corrected_phrases)

print('saving...')
if not os.path.exists('data'):
    os.mkdir('data')

vocab_dir = os.path.join('data', 'cache')
if not os.path.exists(vocab_dir):
    os.mkdir(vocab_dir)

vocab_filename = os.path.join(vocab_dir, args.dataset + '_vocab.pkl')
with open(vocab_filename, 'wb') as f:
    pickle.dump(output_vocab, f, pickle.HIGHEST_PROTOCOL)    

print('identifying similar phrases and writing to disk')
all_tokens = set()
for split in splits:
    for phrase in output_vocab[split + '_corrected']:
        all_tokens.update(phrase.split()) 

wn_lemmas = set([w.replace('_', ' ') for w in wn.all_lemma_names()])
is_flickr = args.dataset == 'flickr'
manual_corrections = {'jacket' : set(['cap', 'case', 'dress']),
                      'hair' : set(['fabric', 'cloth', 'material']),
                      'skirt' : set(['girl', 'ring', 'doll', 'bird'])}

for split in splits:
    corrected_phrases = output_vocab[split + '_corrected']
    corrected_tokens = {}
    for phrase in corrected_phrases:
        corrected_tokens[phrase] = phrase.split()

    hypernym_dictionary = {}
    for i, testphrase in enumerate(corrected_phrases):
        if i % 2500 == 0:
            print('split %s: %i of %i' % (split, i, len(corrected_phrases)))

        hypernym_dictionary[testphrase] = []
        if testphrase == 'unk':
            continue

        # lets gather all hypernyms and synonyms
        tokens = corrected_tokens[testphrase]
        replacement_words = []
        for i, token in enumerate(tokens):
            replacement_words.append(set([token]))
            if token not in wn_lemmas:
                continue

            skip_hypernym = token in ['left', 'right', 'black', 'white', 'brown', 'raft', 'pot']
            syns = wn.synsets(token)
            for syn in syns:
                for word in syn.lemmas():
                    word = word.name().replace('_', ' ').encode('ascii', 'ignore')
                    if word in all_tokens:
                        replacement_words[i].add(word)

                if skip_hypernym:
                    continue

                for hypernym in syn.hypernyms():
                    for word in hypernym.lemmas():
                        word = word.name().replace('_', ' ').encode('ascii', 'ignore')
                        if word in all_tokens and (token not in manual_corrections or word not in manual_corrections[token]):
                            replacement_words[i].add(word)

        candidates = set()
        if is_flickr:
            # also consider subsets of testphrase and replacements of the
            # head word
            copy_tokens = list(tokens)
            for replacement in replacement_words[-1]:
                copy_tokens[i] = replacement
                set_tokens = set(copy_tokens)
                for phrase, words in corrected_tokens.iteritems():
                    if phrase == testphrase or phrase in candidates:
                        continue
                    
                    words = set(words)
                    if len(words.intersection(set_tokens)) == len(words):
                        candidates.add(phrase)

        # now lets handle replacing words
        for phrase, words in corrected_tokens.iteritems():
            if phrase == testphrase or len(words) != len(replacement_words) or phrase in candidates:
                continue

            match = True
            for replacements, word in zip(replacement_words, words):
                match = word in replacements
                if not match:
                    break

            if match:
                candidates.add(phrase)

        hypernym_dictionary[testphrase] = list(candidates)

    corrected_filename = os.path.join(vocab_dir, '%s_%s_corrected.txt' % (args.dataset, split))
    with open(corrected_filename, 'w') as f:
        for phrase in corrected_phrases:
            for w in hypernym_dictionary[phrase]:
                phrase += ',' + w

            f.write(phrase + '\n')

    print('%s %s vocab contains %i entries' % (args.dataset, split, len(corrected_phrases)))

if args.embedding == 'fasttext':
    print('getting subset of fasttext tokens used in the %s dataset' % args.dataset)
    embedding_size = 300
    with open(os.path.join('data', 'cc.en.300.vec'), 'r') as f:
        embedding_dict = {}
        for i, line in enumerate(f):
            if i % 250000 == 0:
                print('Reading embedding number %i' % i)
                    
            line = line.strip()
            if not line:
                continue

            vec = line.split()

            # ignore lines that don't specify embeddings
            if len(vec) != embedding_size + 1:
                continue
            
            label = vec[0].lower()
            if label in all_tokens:
                assert(len(vec) == embedding_size)
                embedding_dict[label] = line

    print('writing embedding dictionary to disk with %i entries' % len(embedding_dict))
    with open(os.path.join('data', '%s_%s_embedding.txt' % (args.dataset, args.embedding)), 'w') as f:
        for word, embedding in embedding_dict.iteritems():
            f.write('%s\n' % embedding)

            
