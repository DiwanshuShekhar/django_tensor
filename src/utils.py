import cPickle
import numpy as np


def build_embedding_matrix(vp_file, embeb_file,
                           vocabulary=None,
                           embed_len=None):
    """
    vocabulary is not None when called by infer function
    :param ids: tensor of ids - shape = [config.MAX_SENTENCE_LEN]
    :param vp_file: Trained VocabularyProcessor object
    :return: tensor shape = [config.MAX_SENTENCE_LEN, embedding_dimension]
    """
    if vocabulary is None:
        with open(vp_file, 'rb') as fh:
            vp = cPickle.load(fh)

        vocabulary = vp.get_vocab()

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
