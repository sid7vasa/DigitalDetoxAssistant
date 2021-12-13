class SkillWord:
    def __init__(self, word, root):
        self.word = word
        self.sentences = []
        self.count = 0
        self.embeddings = []
        self.indices = []
        self.root = root
        self.skills = []

    def __str__(self):
        return str(self.word + str(" [") + str(self.count) + str(" ") + str(set(self.skills)) + str("]"))

    def __repr__(self):
        return self.__str__()

    def update(self, sentence, elmo_vector, index, skill):
        assert len(elmo_vector.shape) == 1
        self.sentences.append(sentence)
        self.indices.append(index)
        self.embeddings.append(elmo_vector)
        self.skills.append(skill)
        self.count = self.count + 1
