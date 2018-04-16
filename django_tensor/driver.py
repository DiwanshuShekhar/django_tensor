from abc import ABCMeta, abstractmethod
import tensorflow as tf


class InferenceDriver:

    __metaclass__ = ABCMeta

    def __init__(self, checkpoint_file=None,
                        tensorflow_model=None,
                        model_type=None):
        self.checkpoint_file = checkpoint_file
        self.tensorflow_model = tensorflow_model
        self.model_type = model_type

    @abstractmethod
    def build_user_input(self):
        """
        Takes tensorflow_model.user_data and does necessary pre-processing
        before the infer call is made
        :param tensorflow_model: instance of model.TensorflowModel
        :return:
        """
        pass

    def infer(self):
        user_input = self.build_user_input()
        print(user_input)
        self.tensorflow_model.set_processed_user_data(user_input)
        predict_op = self.tensorflow_model.get_prediction()
        data_op = self.tensorflow_model.get_processed_user_data()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.checkpoint_file)
            prediction = sess.run(predict_op)
            input_data = sess.run(data_op)
            print(input_data)
            return prediction





