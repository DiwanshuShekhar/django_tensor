import tensorflow as tf
from abc import ABCMeta, abstractmethod


class TensorflowModel:

    __metaclass__ = ABCMeta

    def __init__(self, user_input=None):
        self.raw_user_data = user_input
        self.processed_user_data = None

    def set_processed_user_data(self, processes_user_data):
        self.processed_user_data = processes_user_data

    @abstractmethod
    def _get_user_input(self):
        """
        Takes self.processed_user_data and builds a pipeline to inference
        :return:
        """
        pass


    @abstractmethod
    def _inference(self):
        """
        This is where Neural Network is built
        :return:
        """
        pass

    @abstractmethod
    def _create_predictions(self):
        """
        Takes the value returned from inference and
        predicts the label
        :return:
        """
        pass

    def get_loss(self):
        return tf.reduce_mean(self.losses)

    def get_logits(self):
        return self.logits

    def get_prediction(self):
        self._get_user_user_input()
        self._inference()
        self._create_predictions()
        return self.predicted_labels
