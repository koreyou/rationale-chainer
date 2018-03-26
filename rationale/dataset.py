# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import logging

import nltk
import numpy as np
from chainer.datasets import DictDataset

nltk.download(info_or_id='punkt')  # need it before importing nltk.tokenize

from nltk.tokenize import word_tokenize

from rationale.word_embedding import load_word_embedding
logger = logging.getLogger(__name__)


def prepare_data(train_path, word2vec_path, test_path=None):
    logger.info("Preparing data")

    logger.info("Loading word embedding")
    w2v, vocab = load_word_embedding(word2vec_path, max_vocab=100000)

    logger.info("Creating dataset")
    train_dataset = _read_beer_dataset(train_path, vocab, max_tokens=50)

    if len(test_path) > 0:
        test_dataset = _read_beer_dataset(test_path, vocab, max_tokens=50)
    else:
        test_dataset = None

    return w2v, vocab, train_dataset, test_dataset



def _read_beer_dataset(path, vocab, max_tokens):
    def tokenize(text):
        words = []
        for i, w in enumerate(word_tokenize(text)):
            if i >= max_tokens:
                break
            words.append(vocab.get(w, vocab['<unk>']))
        return words

    xs = []
    appearance_scores = []
    aroma_scores = []
    overall_scores = []
    palate_scores = []
    taste_scores = []

    with open(path) as fin:
        itr = iter(fin)
        while True:
            next(itr)  # beer/name
            next(itr)  # beer/beerId
            next(itr)  # beer/ABV
            next(itr)  # beer/style
            for line in itr:
                if len(line.strip()) > 0:
                    break
            else:
                break
            assert line.startswith("review/appearance: ")
            appearance = float(line[19:].strip())

            line = next(itr)
            assert line.startswith("review/aroma: ")
            aroma = float(line[14:].strip())

            line = next(itr)
            assert line.startswith("review/palate: ")
            palate = float(line[15:].strip())

            line = next(itr)
            assert line.startswith("review/taste: ")
            taste = float(line[14:].strip())

            line = next(itr)
            assert line.startswith("review/overall: ")
            overall = float(line[15:].strip())

            line = next(itr)
            assert line.startswith("review/text: ")
            text = line[13:].strip()
            tokens = tokenize(text)
            if len(tokens) == 0:
                continue
            xs.append(np.array(tokens, np.int32))
            appearance_scores.append(appearance)
            aroma_scores.append(aroma)
            overall_scores.append(overall)
            palate_scores.append(palate)
            taste_scores.append(taste)
    return {
        'appearance': DictDataset(
            xs=xs, ys=np.array(appearance_scores, dtype=np.float32)),
        'aroma': DictDataset(
            xs=xs, ys=np.array(aroma_scores, dtype=np.float32)),
        'overall': DictDataset(
            xs=xs, ys=np.array(overall_scores, dtype=np.float32)),
        'palate': DictDataset(
            xs=xs, ys=np.array(palate_scores, dtype=np.float32)),
        'taste': DictDataset(
            xs=xs, ys=np.array(taste_scores, dtype=np.float32)),
    }
