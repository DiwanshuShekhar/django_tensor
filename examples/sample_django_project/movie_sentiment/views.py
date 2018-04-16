# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

from django_tensor.model import TensorflowModel
from django_tensor.driver import InferenceDriver
import utils

import tensorflow as tf
import logging
import time
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('model')

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# create your model


class SentimentModel(TensorflowModel):

    def __init__(self, raw_user_input):
        TensorflowModel.__init__(self, raw_user_input)

        # hyper-parameters
        self.n_neurons = 300
        self.learning_rate = 0.001

    # implement abstract methods
    def _get_user_input(self):
        """
        Takes self.processed_user_data and builds a pipeline to inference
        :return:
        """
        self.review = tf.reshape(self.processed_user_data[0], [1, 160, 200])
        self.review_len = tf.reshape(self.processed_user_data[1], [1])
        logging.info("Shape of review {}".format(self.review))
        logging.info("Shape of review len {}".format(self.review_len.shape))

    def _inference(self):
        """
        This is where Neural Network is built. Should be the same as what was used in training the model
        :return:
        """
        try:
            with tf.variable_scope('lstm', reuse=True):
                cell_context = tf.nn.rnn_cell.LSTMCell(self.n_neurons, forget_bias=2.0,
                                                       use_peepholes=True, state_is_tuple=True)
                outputs, output_states = tf.nn.dynamic_rnn(cell_context,
                                                           self.review,
                                                           dtype=tf.float32,
                                                           sequence_length=self.review_len)
                self.logits = tf.norm(output_states.h, axis=1)
        except ValueError:
            with tf.variable_scope('lstm'):
                cell_context = tf.nn.rnn_cell.LSTMCell(self.n_neurons, forget_bias=2.0,
                                                       use_peepholes=True, state_is_tuple=True)
                outputs, output_states = tf.nn.dynamic_rnn(cell_context,
                                                           self.review,
                                                           dtype=tf.float32,
                                                           sequence_length=self.review_len)
                self.logits = tf.norm(output_states.h, axis=1)

    def _create_predictions(self):
        """
        Takes the value returned from inference and
        predicts the label
        :return:
        """
        print(self.logits)
        probabilities = tf.sigmoid(self.logits)
        predicted_labels = tf.greater_equal(probabilities, 0.85)
        return tf.cast(predicted_labels, tf.int64)

    def get_processed_user_data(self):
        return self.processed_user_data

# create your driver


class SentimentDriver(InferenceDriver):

    def __init__(self, checkpoint_file=None,
                       tensorflow_model=None,
                       model_type=None):
        InferenceDriver.__init__(self, checkpoint_file, tensorflow_model, model_type)

    def build_user_input(self):
        """
        Takes tensorflow_model.user_data and does necessary pre-processing
        before the infer call is made
        :param tensorflow_model: instance of model.TensorflowModel
        :return:
        """
        vocabulary = utils.get_vocab(os.path.join(CURRENT_DIR, 'vocabulary.json'))  # can be added to settings.py for
        # performance
        logging.info("Length of vocabulary {}".format(len(vocabulary)))
        features = {}
        features['review'], features['review_len'] = utils.get_ids_from_string(self.tensorflow_model.raw_user_data,
                                                                               160,
                                                                               vocabulary)
        logging.info("ids are {}".format(features['review']))

        start = time.time()
        embeddings_mat = utils.build_embedding_matrix(os.path.join(CURRENT_DIR, 'glove.6B.200d.txt'),
                                                      vocabulary=vocabulary,
                                                      embed_len=200)  # can be added to settings.py for performance
        logging.info("Elapsed time to build embedding matrix {}".format(time.time() - start))
        logging.info("Shape of embedding matrix {}".format(embeddings_mat.shape))

        start = time.time()
        features['review'] = tf.nn.embedding_lookup(embeddings_mat, features['review'])
        logging.info("Elapsed time for embedding lookup {}".format(time.time() - start))
        return features['review'], features['review_len']

# Create your views here.


@csrf_exempt
def q(request):
    """
    Example API call for this view function assuming the server is running at localhost:8000
    http://127.0.0.1:8000/movie_sentiment/q/?user-stmt=I%20love%20this%20movie.%20The%20director%20did%20a%20great%20job
    :param request:
    :return:
    """
    if request.method == "POST":
        myQueryDict = request.POST
    else:
        myQueryDict = request.GET

    review = myQueryDict.__getitem__("user-stmt")

    model = SentimentModel(review)
    driver = SentimentDriver(checkpoint_file=os.path.join(CURRENT_DIR, 'checkpoints/390.ckpt'),
                             tensorflow_model=model)
    start_time = time.time()
    prediction = driver.infer()
    end_time = time.time()

    result = {}
    result['prediction'] = prediction[0]
    result['responseTime'] = end_time - start_time
    jr = json.dumps(result)

    return HttpResponse(jr)


