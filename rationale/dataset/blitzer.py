# -*- coding: utf-8 -*-
"""
Module that loads data distributed at
https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

The dataset was presented on the following paper:

J. Blitzer, M. Dredze, F. Pereira. 2007. Biographies, Bollywood, Boom-boxes and
Blenders: Domain Adaptation for Sentiment Classification. ACL.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging
import math
import os
import codecs

import nltk

nltk.download(info_or_id='punkt')  # need it before importing nltk.tokenize

import numpy as np
from nltk.tokenize import word_tokenize
from chainer.datasets import split_dataset_random

from rationale.word_embedding import load_word_embedding
from rationale.dataset.common import create_dataset

logger = logging.getLogger(__name__)


_DOMAIN_LIST = [
    "apparel", "automotive", "baby", "beauty", "books", "camera_&_photo",
    "cell_phones_&_service", "computer_&_video_games", "dvd", "electronics",
    "gourmet_food", "grocery", "health_&_personal_care", "jewelry_&_watches",
    "kitchen_&_housewares", "magazines", "music", "musical_instruments",
    "office_products", "outdoor_living", "software", "sports_&_outdoors",
    "tools_&_hardware", "toys_&_games", "video"]


_RATIO = {'train': 0.7, 'validation': 0.1, 'test': 0.2}
_SEED = 418


def read_amazon_reviews(base_path, vocab, max_tokens=10000):
    texts = []
    labels = []
    domains = []
    for d, domain_name in enumerate(_DOMAIN_LIST):
        for l, filename in enumerate(("negative.review", "positive.review")):
            # negative is 0, positive is 1
            path = os.path.join(base_path, domain_name, filename)
            for t in read_single_review(path, vocab, max_tokens):
                texts.append(t)
                labels.append(l)
                domains.append(d)
    labels = np.asarray(labels, np.int32)
    domains = np.asarray(domains, np.int32)
    return create_dataset(texts, labels, domains)


def prepare_blitzer_data(base_path, word2vec_path):
    logger.info("Preparing data")

    logger.info("Loading word embedding")
    w2v, vocab = load_word_embedding(word2vec_path, max_vocab=100000)

    logger.info("Creating dataset")
    # Read all dataset
    dataset = read_amazon_reviews(base_path, vocab, max_tokens=50)
    train, val, test = _split_dataset(dataset)
    # Reverse domain_dict
    domain_dict = {i: d for i, d in enumerate(_DOMAIN_LIST)}
    label_inv_dict = {0: "neg", 1: "pos"}

    return w2v, vocab, train, val, test, label_inv_dict, domain_dict


def _split_dataset(dataset):
    n_train = int(math.floor(len(dataset) * _RATIO['train']))
    n_validation = int(math.floor(len(dataset) * _RATIO['validation']))
    train, rest = split_dataset_random(dataset, n_train, seed=_SEED)
    val, test = split_dataset_random(rest, n_validation, seed=_SEED)
    assert len(train) + len(val) + len(test) == len(dataset)
    return train, val, test


def read_single_review(path, vocab, max_tokens):
    is_review = False
    with codecs.open(path, encoding='latin-1') as fin:
        for line in fin:
            line = line.strip()
            if line == '<review_text>':
                assert not is_review
                words = []
                is_review = True
            elif line == '</review_text>':
                assert is_review
                if len(words) > 0:
                    yield np.array(words, np.int32)
                is_review = False
            elif is_review and len(words) < max_tokens:
                for i, w in enumerate(word_tokenize(line)):
                    if len(words) >= max_tokens:
                        break
                    words.append(vocab.get(w, vocab['<unk>']))
