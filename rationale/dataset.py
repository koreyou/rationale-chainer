# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import gzip
import logging

import numpy as np
from chainer.datasets import DictDataset

from rationale.word_embedding import load_word_embedding

logger = logging.getLogger(__name__)


def prepare_data(train_path, word2vec_path, aspect, test_path=None):
    logger.info("Preparing data")

    logger.info("Loading word embedding")
    w2v, vocab = load_word_embedding(word2vec_path, max_vocab=100000)

    logger.info("Creating dataset")
    train_dataset = _read_beer_dataset(train_path, aspect, vocab, max_tokens=50)

    if len(test_path) > 0:
        test_dataset = _read_beer_dataset(
            test_path, aspect, vocab, max_tokens=50)
    else:
        test_dataset = None

    return w2v, vocab, train_dataset, test_dataset



def _read_beer_dataset(path, aspect, vocab, max_tokens):
    xs = []
    scores = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path) as fin:
        for line in fin:
            s, words = line.strip().split("\t")
            s = list(map(float, s))
            tokens= [vocab.get(w, vocab['<unk>']) for w in words[:max_tokens]]
            if len(tokens) == 0:
                continue
            xs.append(np.array(tokens, np.int32))
            scores.append(s[aspect])

    return DictDataset(xs=xs, ys=np.array(scores, dtype=np.float32))
