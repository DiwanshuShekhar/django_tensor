import tensorflow as tf
from abc import ABC, abstractmethod

class TensorflowModel(ABC):

    def __init__(self, user_input=None):
        self.user_data = user_input
        super().__init__()

    @abstractmethod
    def _get_user_input(self):

    @abstractmethod
    def _inference(self):

    @abstractmethod 
    def _create_predictions(self):

    def get_loss(self):
        return tf.reduce_mean(self.losses)

    def get_logits(self):
        return self.logits
        
    def get_prediction(self):
        self._get_user_user_input()
        self._inference()
        self._create_predictions()
        return self.predicted_labels
