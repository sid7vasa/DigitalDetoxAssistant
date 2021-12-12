import numpy as np
import config
import spacy
import nltk
import re


def tokenize(sentence):
    tokens = nltk.word_tokenize(sentence)
    return tokens


def remove_punctuation(text):
    try:
        punctuation_free = re.sub(r'[^\w\s]', '', text)
        punctuation_free = punctuation_free.replace("\n", " ")

    except:
        return " "
    return punctuation_free


def sentence_tokenization(text):
    tokens = nltk.sent_tokenize(text)
    return tokens


def lower_case(text):
    try:
        lower = text.lower()
    except:
        print("Lower Case failed?")
        return ""
    return lower


def find_word(li, word):
    for i, el in enumerate(li):
        if word == el.word:
            return i
    return -1


def find_word_root(li, word):
    for i, el in enumerate(li):
        if word == el.root:
            return i
    return -1


class Utils:
    def __init__(self):
        self.bert_vocab = list(np.load("../data/bert_vocab.npy"))
        self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    def is_rare(self, word):
        if self.lemmatize(word) in self.bert_vocab and self.lemmatize(word) is not '-PRON-':
            index = self.bert_vocab.index(self.lemmatize(word))
            if index >= config.rare_word_threshold and self.lemmatize(word):
                return True
            else:
                return False
        elif self.lemmatize(word) == '-PRON-':
            return False
        else:
            return True

    def lemmatize(self, text):
        texts = [text]
        output = []
        for i in texts:
            s = [token.lemma_ for token in self.nlp(i)]
            output.append(' '.join(s))
        return output[0]


def extract_text(tag):
    if tag is not None:
        li = [text.text for text in tag]
        return ". ".join(li)


def preprocess_sentence_text(text):
    text = lower_case(text)
    sentences = sentence_tokenization(text)
    return [remove_punctuation(sent) for sent in sentences]
