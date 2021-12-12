from utils import extract_text
from utils import preprocess_sentence_text
from utils import tokenize
import numpy as np
import textwrap


class Webpage:
    def __init__(self, url, driver, time=None, title=None):
        self.url = url
        self.time = time
        self.title = title
        self.fetched = self.fetch_url(driver)
        self.count = 0
        self.processed_fetch = self.process_fetched()

    def fetch_url(self, driver):
        driver.get(self.url)
        #  extracts text from selenium objects and joins them with "." so that it can later be sentence tokenized.
        fetched = {}
        try:
            fetched['h1'] = extract_text(driver.find_elements_by_tag_name("h1"))
        except:
            fetched['h1'] = []

        try:
            fetched['h2'] = extract_text(driver.find_elements_by_tag_name("h2"))
        except:
            fetched['h2'] = []

        try:
            fetched['h3'] = extract_text(driver.find_elements_by_tag_name("h3"))
        except:
            fetched['h3'] = []

        try:
            fetched['h4'] = extract_text(driver.find_elements_by_tag_name("h4"))
        except:
            fetched['h4'] = []

        try:
            fetched['h5'] = extract_text(driver.find_elements_by_tag_name("h5"))
        except:
            fetched['h5'] = []

        try:
            fetched['h6'] = extract_text(driver.find_elements_by_tag_name("h6"))
        except:
            fetched['h6'] = []

        try:
            fetched['span'] = extract_text(driver.find_elements_by_tag_name("span"))
        except:
            fetched['span'] = []

        try:
            fetched['p'] = extract_text(driver.find_elements_by_tag_name("p"))
        except:
            fetched['p'] = []

        return fetched

    def process_fetched(self):
        processed_dict = {}
        for i, tag in enumerate(self.fetched.keys()):
            under_50_tag_sentences = []
            tag_sentences = preprocess_sentence_text(self.fetched[tag])
            # gives list of sentences which are lower case, removed punctuation.
            # split if any of these sentences are more than length 50.
            for sentence in tag_sentences:
                words = tokenize(sentence)
                self.count = self.count + len(words)
                if len(words) < 50:
                    under_50_tag_sentences.append(sentence)
                else:
                    # ToDo splitting the sentence into characters not words, fix it later
                    split_sentence = textwrap.wrap(sentence, 100)

                    under_50_tag_sentences.extend(split_sentence)
            processed_dict[tag] = under_50_tag_sentences
        return processed_dict
