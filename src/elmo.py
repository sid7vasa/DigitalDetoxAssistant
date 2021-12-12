import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import config


class Elmo:
    def __init__(self, path=config.ELMO_PATH):
        print("Using real Elmo?")
        self.elmo = hub.load(path)

    def run(self, sentence):
        try:
            if isinstance(sentence, str):
                embeddings = self.elmo.signatures["default"](tf.constant([sentence]))["elmo"]
                return np.array(embeddings)
            elif isinstance(sentence, list):
                embeddings = self.elmo.signatures["default"](tf.constant(sentence))["elmo"]
                return np.array(embeddings)
        except ValueError:
            print("Word not found in sentence")
            return np.array([])
