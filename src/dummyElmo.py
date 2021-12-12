import numpy as np


class Elmo:
    def __init__(self, path=None):
        print("Not using elmo!")
        pass

    def run(self, sentence):
        try:
            if isinstance(sentence, str):
                embeddings = np.ones((1, 500, 1024))
                return np.array(embeddings)
            elif isinstance(sentence, list):
                embeddings = np.ones((len(sentence), 500, 1024))
                return np.array(embeddings)
        except ValueError:
            print("Word not found in sentence")
            return np.array([])
