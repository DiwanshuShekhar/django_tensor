try:
    from data import VocabProcessor, tokenizer_fn
except:
    from .data import VocabProcessor, tokenizer_fn
import config
import tensorflow as tf
import numpy as np
import cPickle
import os
import json
import time
try:
    from .model import SentimentModel
except:
    from model import SentimentModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_embedding_matrix(vp_file, embeb_file, vocabulary=None):
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
    
    logger.info("Length of vacabulary: {}".format(len(vocabulary)))

    embeddings_mat = np.random.uniform(-0.25, 0.25, (len(vocabulary), config.EMBED_LEN)).astype("float32")

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


def parse_input(example):
    """

    :return: dict
    """
    features = tf.parse_single_example(example, features={
                                           'review': tf.FixedLenFeature([config.MAX_SENTENCE_LEN], tf.int64),
                                           'review_len': tf.FixedLenFeature([], tf.int64),
                                           'label': tf.FixedLenFeature([], tf.int64),
                                   })
    try:
        with tf.variable_scope('embedding', reuse=True):
            embeddings_mat = tf.get_variable("word_embeddings")
            logging.info("Reused embedding")
    except ValueError:
        logging.info("Getting embedding for the first time")
        with tf.variable_scope('embedding'):
            embeddings_matrix = build_embedding_matrix(os.path.join(BASE_DIR, config.VOCAB_PROCESSOR), 
                                                       os.path.join(BASE_DIR, config.EMBED_FILE))
            embeddings_mat = tf.get_variable("word_embeddings", trainable=False, initializer=embeddings_matrix)
    print("shape of embedding matrix", embeddings_mat.shape)  #
    print("list of feature['review'] indices", features['review'])

    features['review'] = tf.nn.embedding_lookup(embeddings_mat, features['review'])
    return features['review'], features['label'], features['review_len']


def build_input_pipeline(in_files, batch_size, num_epochs=None):
    """
    Build an input pipeline with the DataSet API
    :param in_files list of tfrecords filenames
    :return dataset iterator (use get_next() method to get the next batch of data from the dataset iterator
    """
    dataset = tf.contrib.data.TFRecordDataset(in_files)
    dataset = dataset.map(parse_input, num_threads=12,
                          output_buffer_size=10 * batch_size)  # Parse the record to tensor
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.batch(batch_size)
    if num_epochs:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat()  # Repeat the input indefinitely.
    iterator = dataset.make_initializable_iterator()
    return iterator


def train():
    """
    Builds the graph and runs the graph in a session
    :return:
    """
    train_files = config.TRAIN_FILES
    validation_files = config.VALIDATION_FILES

    with tf.Graph().as_default():

        logging.info("Building train input pipeline")
        input_train_iter = build_input_pipeline(train_files, config.TRAIN_BATCH_SIZE,
                                                num_epochs=config.NUM_EPOCHS)

        logging.info("Building validation input pipeline")
        input_validation_iter = build_input_pipeline(validation_files, config.VALIDATION_BATCH_SIZE)

        model = SentimentModel(input_train_iter, input_validation_iter)

        logging.info("Building graph")
        train_op = model.build_graph()

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(input_train_iter.initializer)
        sess.run(input_validation_iter.initializer)

        saver = tf.train.Saver()

        #  starting the training
        logging.info("Training starts...")
        batch = 0
        epoch_step = 0
        num_batches = int(50000/config.TRAIN_BATCH_SIZE)
        try:
            while True:
                # here calculate accuracy and/or training loss
                _, loss = sess.run([train_op, model.get_loss()])
                logger.info("Train batch step {0}: loss = {1}".format(batch, loss))
                lgts = sess.run(model.get_logits())
                logger.info("Train batch step {0}: logits = {1}".format(batch, lgts))
                acc = sess.run(model.get_validation_accuracy())
                logger.info("Train batch step {0}: Acc = {1}".format(batch, acc))
                batch += 1
                if batch % num_batches == 0:  # completes an epoch examples/batch_size
                    logger.info("Completed epoch {0}".format(epoch_step))
                    # save a model checkpoint
                    saver.save(sess, './checkpoints/{}.ckpt'.format(batch))
                    epoch_step += 1
        except tf.errors.OutOfRangeError:
            logging.info('Done training for {0} epochs, {1} steps.'.format(epoch_step, batch))

        sess.close()


def build_review(review):
    features = {}
    with open(os.path.join(BASE_DIR, config.VOCABULARY), 'r') as fh:
        vocabulary = json.load(fh)

    vp = VocabProcessor(config.MAX_SENTENCE_LEN)
    vp.set_vocab(vocabulary)
    features['review'], features['review_len'] = vp.get_ids_from_string(review)
    logging.info("Ids for provided review: {}".format(features['review']))
    logging.info("Length of review: {}".format(features['review_len']))

    try:
        with tf.variable_scope('embedding', reuse=True):
            embeddings_mat = tf.get_variable("word_embeddings")
            logging.info("Reused embedding")
    except ValueError:
        logging.info("Getting embedding for the first time")
        with tf.variable_scope('embedding'):
            embeddings_matrix = build_embedding_matrix(os.path.join(BASE_DIR, config.VOCAB_PROCESSOR), 
                                                       os.path.join(BASE_DIR, config.EMBED_FILE),
                                                       vocabulary = vp.get_vocab())
            embeddings_mat = tf.get_variable("word_embeddings", trainable=False, initializer=embeddings_matrix)
    print("shape of embedding matrix", embeddings_mat.shape)  #
    print("list of feature['review'] indices", features['review'])

    features['review'] = tf.nn.embedding_lookup(embeddings_mat, features['review'])
    return features['review'], features['review_len']


def infer(review):
    
    logging.info("Building user review")
    review = build_review(review)
    print(review)

    model = SentimentModel(None, None, review=review)

    logging.info("Building graph")
    predict_op = model.get_sentiment()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, os.path.join(BASE_DIR, config.CHECKPOINT_FILE))
        sentiment = sess.run(predict_op)
        logger.info("Sentiment = {0}".format(sentiment))
        return sentiment


def infer_app(vocab_proc, embeddings_mat, review):
    """
    Same as infer but optimized to be called by views.py in Django Apps
    """
    features = {}
    features['review'], features['review_len'] = vocab_proc.get_ids_from_string(review)
   
    start = time.time()
    features['review'] = tf.nn.embedding_lookup(embeddings_mat, features['review'])
    logging.info("Time taken for embedding lookup {}".format(time.time() - start))

    review = features['review'], features['review_len']
    
    model = SentimentModel(None, None, review=review)

    logging.info("Building graph")
    start = time.time()
    predict_op = model.get_sentiment()
    logging.info("Time taken to build graph {}".format(time.time() - start))
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        logging.info("Time taken to start a session {}".format(time.time() - start))
        start = time.time()
        saver.restore(sess, os.path.join(BASE_DIR, config.CHECKPOINT_FILE))
        logging.info("Time taken to restore checkpoint {}".format(time.time() - start))
        start = time.time()
        sentiment = sess.run(predict_op)
        logging.info("Time taken to predict op in th session{}".format(time.time() - start))
        logger.info("Sentiment = {0}".format(sentiment))
        return sentiment


if __name__ == "__main__":
    """
    ip = build_input_pipeline('./dataset/train.tfrecords', 50, 1)
    data = ip.get_next()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(ip.initializer)
    data_res = sess.run(data)
    print(data_res[0], data_res[1], data_res[2])
    sess.close()
    """

    """
    #  load embedding matrix
    m = build_embedding_matrix(config.VOCAB_PROCESSOR, config.EMBED_FILE)
    print("m shape", m.shape)
    print(m)
    """
    #train()
    review = "It nicely predicted the conditioning of human minds with that of the patient. " \
            "We all believe that what we believe is true with our own point of view and we want " \
            "to solve all the problem accordingly."
    infer(review)


