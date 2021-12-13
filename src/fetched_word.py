class FetchedWord:
    def __init__(self, word, root, tag=None):
        self.word = word
        self.sentences = []
        self.count = 0
        self.embeddings = []
        self.indices = []
        self.root = root
        self.tag = tag

    def __str__(self):
        return str(self.word + str(str(" [") + str(self.count) + str(" ") + str(self.tag)) + str("]"))

    def __repr__(self):
        return self.__str__()

    def update(self, sentence, elmo_vector, index):
        assert len(elmo_vector.shape) == 1
        self.sentences.append(sentence)
        self.indices.append(index)
        self.embeddings.append(elmo_vector)
        self.count = self.count + 1
