import cPickle
import numpy as np
import json


def get_vocab(vocab_file):
    """
    :param vocab_file: string path to a json file of word and its id
    :return: dict of word and id
    """
    with open(vocab_file, 'r') as fh:
        vocabulary = json.load(fh)

    return vocabulary


def get_ids_from_string(review, max_sentence_length, vocabulary):
    """
    converts a user provided review to a list of ids
    :param review: string
    :return: tuple of list of ids and length of user provided review
    """
    ids = [1] * max_sentence_length
    words = review.split()
    for i, word in enumerate(words):
        if i > max_sentence_length - 1:
            break
        ids[i] = vocabulary.get(word, 0)
    return ids, len(words)


def build_embedding_matrix(embeb_file,
                           vocabulary=None,
                           embed_len=None):
    """
    :param embed_file: string path to embedding file
    :param vocabulary:
    :return: tensor shape = [MAX_SENTENCE_LEN, embedding_dimension]
    """
    embeddings_mat = np.random.uniform(-0.25, 0.25, (len(vocabulary), embed_len)).astype("float32")

    embed_dict = {}
    with open(embeb_file, 'r') as fh:
        for line in fh:
            tokens = line.split(" ")
            embed_word, embed_vector = tokens[0], tokens[1:]
            embed_dict[embed_word] = embed_vector

    for word, id in vocabulary.items():
        if word in embed_dict:
            embeddings_mat[id] = embed_dict[word]

    del embed_dict
    return embeddings_mat
