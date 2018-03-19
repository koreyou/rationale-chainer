# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import six
from word_embedding_loader import WordEmbedding

UNK_ID_CANDIDATES = [b'<unk>', b'<UNK>']


def infer_unk_id(vocab):
    """

    Args:
        vocab (dict): Mapping from words (bytes) to index (int)

    Returns:
        bytes: "unk" id (as a word in ``vocab``)

    """
    candidates = []
    for c in UNK_ID_CANDIDATES:
        if c in vocab:
            candidates.append(c)
    if len(candidates) > 1:
        raise ValueError(b'Ambiguous unk id found (' + bytes(candidates) + b')')
    if len(candidates) == 0:
        return None
    return candidates[0]


def insert_vocab(vocab, vectors, key, index, vec):
    """
    Insert new word to word embedding.

    .. warning:: This function has side effect on ``vocab``.

    Args:
        vocab (dict): Mapping from word (bytes) to index (int)
        vectors (numpy.ndarray): Word embedding vectors
        key (bytes): New word to insert to move
        index (int): Index to which ``vec`` is inserted
        vec (numpy.ndarray): A vector to insert

    Returns:
        dict: Modified vocab file
        numpy.ndarray: Modified vectors

    """
    assert vec.ndim == 1 or (vec.ndim == 2 and vec.shape[0] == 1)
    assert vec.shape[-1] == vectors.shape[1]
    if key in vocab:
        raise KeyError(b'key "' + key + b'"%s already exists in vocab.')

    for k in vocab.keys():
        if vocab[k] >= index:
            vocab[k] += 1

    vocab[key] = index
    vectors = np.insert(vectors, index, vec, axis=0)
    return vocab, vectors


def move_vocab(vocab, vectors, key, index):
    """
    Move the particular word to specified index.

    .. warning:: This function has side effect on ``vocab``.

    Args:
        vocab (dict): Mapping from word (bytes) to index (int)
        vectors (numpy.ndarray): Word embedding vectors
        key (bytes): New word to insert to move
        index (int): Index to which ``vec`` is inserted

    Returns:
        dict: Modified vocab file
        numpy.ndarray: Modified vectors
        int or None: Original index of the key in vocab

    """
    if key not in vocab:
        raise KeyError(b'key "' + key + b'" does no exist in vocab.')
    vec = vectors[vocab[key]]
    vectors = np.delete(vectors, vocab[key], axis=0)
    vectors = np.insert(vectors, index, vec, axis=0)

    old_ind = vocab[key]
    if old_ind == index:
        return vocab, vectors, index
    elif old_ind > index:
        for k in vocab.keys():
            if vocab[k] >= index and vocab[k] < old_ind:
                vocab[k] += 1
    else:
        for k in vocab.keys():
            if vocab[k] > old_ind and vocab[k] <= index:
                vocab[k] -= 1
    vocab[key] = index
    return vocab, vectors, old_ind


def create_unk_least_common(we, n):
    """

    Args:
        we (~WordEmbedding):

    Returns:

    """
    if len(we.vectors) < n:
        raise ValueError('len(we.vectors) < n (%d < %d)' % (len(we.vectors), n))
    if we.freqs is None:
        return np.average(we.vectors[-n:], axis=0)
    else:
        freqs = sorted(six.iteritems(we.freqs), key=lambda k_v: k_v[1])
        return np.average([we.vectors[we.vocab[k]] for k, _ in freqs[:n]],
                             axis=0)


def resize(word_emb, size):
    """
    Reduce number of vocabulary in place.

    Args:
        word_emb (WordEmbedding): word embeddning to resize
        size (int): new size

    Returns:
        ~WordEmbedding: Returns reference to word_emb
    """
    if size < len(word_emb):
        n = len(word_emb) - size
        if word_emb.freqs is not None:
            del_keys = []
            del_inds = []
            for k, v in sorted(
                    six.iteritems(word_emb.freqs), key=lambda k_v: k_v[1])[:n]:
                del_inds.append(word_emb.vocab[k])
                del_keys.append(k)
            n = 0
            del_keys = set(del_keys)
            for k, _ in sorted(
                    six.iteritems(word_emb.vocab), key=lambda k_v: k_v[1]):
                word_emb.vocab -= n
                if k in del_keys:
                    n += 1
                    del word_emb.vocab[k]
                    del word_emb.freqs[k]
        else:
            del_inds = []
            for k, v in list(six.iteritems(word_emb.vocab)):
                if v >= size:
                    del_inds.append(v)
                    del word_emb.vocab[k]
        assert len(del_inds) == n
        word_emb.vectors = np.delete(word_emb.vectors, del_inds, axis=0)

    return word_emb


def load_word_embedding(path, dtype=np.float32, max_vocab=None, unk=b'<unk>'):
    """
    Load pretrained word embedding from a file.
    Args:
        path (str): Path of file to load.
        vocab (str or set or None): Path to vocab files or set of vocab to use.
            Refer
            :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`
            for details.
        dtype (numpy.dtype): Element data type to use for the array.
        max_vocab (int): Number of vocabulary to read.
        unk (bytes or None): The vocabulary for out-of-vocabulary words.
            If ``None`` it will not do any post-precessings to gurentee that
            it exists.

    Returns:
        vectors (numpy.ndarray): Word embedding vectors in shape of
            ``(vocabulary size, feature dimension)``.
        vocab (dict): Mapping from words (bytes) to vector indices (int)
    """
    unk_index = 0
    # just load everything
    we = WordEmbedding.load(path, vocab=None, dtype=dtype, max_vocab=None)

    if unk is not None:
        _unk = infer_unk_id(we.vocab)
        if _unk is None:
            # Create unk
            v = create_unk_least_common(we, 10)
            we.vocab, we.vectors = insert_vocab(
                we.vocab, we.vectors, unk, unk_index, v)
        else:
            we.vocab, we.vectors, _ = move_vocab(
                we.vocab, we.vectors, _unk, unk_index)
            del we.vocab[_unk]
        we.vocab[unk] = unk_index
        if we.freqs is not None:
            max_freq = max(we.freqs.values())
            we.freqs[unk] = max_freq + 1

    if max_vocab is not None:
        # index of eos/unk is always smaller than max_vocab so it is safe to
        # call resize
        resize(we, max_vocab)
    vocab = {k.decode('utf-8'): v for k, v in six.iteritems(we.vocab)}
    return we.vectors, vocab
