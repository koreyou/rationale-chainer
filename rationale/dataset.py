# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import codecs
import gzip
import json
import logging

import numpy as np
from chainer.datasets import DictDataset

from rationale.word_embedding import load_word_embedding

logger = logging.getLogger(__name__)


def prepare_data(train, word2vec_path, aspect, test=None, annotation=None):
    logger.info("Preparing data")

    logger.info("Loading word embedding")
    w2v, vocab = load_word_embedding(word2vec_path, max_vocab=100000)

    logger.info("Creating dataset")
    if train is not None:
        train = _read_beer_dataset(train, aspect, vocab, max_tokens=100)

    if test is not None:
        test = _read_beer_dataset(test, aspect, vocab, max_tokens=50)

    if annotation is not None:
        annotation = _read_annotation(
            annotation, aspect, vocab, max_tokens=10000000)

    return w2v, vocab, train, test, annotation


def _read_beer_dataset(path, aspect, vocab, max_tokens):
    xs = []
    scores = []
    fopen = gzip.open if path.endswith(".gz") else codecs.open
    with fopen(path, mode='rt', encoding='utf-8') as fin:
        for line in fin:
            s, words = line.strip().split("\t")
            s = list(map(float, s.split(" ")))
            tokens= [vocab.get(w, vocab['<unk>'])
                     for w in words.split(" ")[:max_tokens]]
            if len(tokens) == 0:
                continue
            xs.append(np.array(tokens, np.int32))
            scores.append(s[aspect])

    return DictDataset(xs=xs, ys=np.array(scores, dtype=np.float32))


def _read_annotation(path, aspect, vocab, max_tokens):
    xs = []
    scores = []
    intervals = []
    with open(path) as fin:
        for line in fin:
            d = json.loads(line.strip())
            words = d['x']
            tokens = [vocab.get(w, vocab['<unk>']) for w in words[:max_tokens]]
            if len(tokens) == 0:
                continue
            s = d['y']
            xs.append(np.array(tokens, np.int32))
            scores.append(s[aspect])
            intervals.append(d[str(aspect)])

    return DictDataset(
        xs=xs, ys=np.array(scores, dtype=np.float32), intervals=intervals)
