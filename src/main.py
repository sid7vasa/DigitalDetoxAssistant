import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from utils import tokenize
from utils import Utils
from utils import find_word
from utils import find_word_root
import utils
from fetched_word import FetchedWord
import config
from webpage import Webpage

if config.WITH_ELMO:
    import elmo
else:
    import dummyElmo

if __name__ == '__main__':
    options = Options()
    options.headless = config.HEADLESS
    options.add_argument("--window-size=1920,1200")
    driver = webdriver.Chrome(executable_path=config.DRIVER_PATH, options=options)
    if config.WITH_ELMO:
        elmo = elmo.Elmo()
    else:
        elmo = dummyElmo.Elmo()

    if config.DEBUG: print("loading skill words from: ", config.SKILL_WORD_PATH)
    tools = Utils()
    print("Opening your skill information: ")
    with open(config.SKILL_WORD_PATH, 'rb') as handle:
        skill_rare_words = pickle.load(handle)


    def process_word(word):
        word = utils.remove_punctuation(word)
        word = utils.lower_case(word)
        word = tools.lemmatize(word)
        return word


    def process_set(all_words):
        li = []
        for word in all_words:
            li.append(process_word(word))
        return list(set(li))


    def test_url(url):
        webpage = Webpage(url, driver)
        fetched = webpage.process_fetched()
        fetched_rare_words = []
        all_words = []
        print("Processing Text from the web page")
        for tag in fetched.keys():
            if len(fetched[tag]) == 0:
                continue
            embeddings = elmo.run(fetched[tag])
            for i, sentence in enumerate(fetched[tag]):
                words = tokenize(sentence)
                all_words.extend(words)
                for word in words:
                    if tools.is_rare(word):
                        index = words.index(word)
                        word_ind = find_word(fetched_rare_words, word)
                        if word_ind >= 0:
                            fetched_rare_words[word_ind].update(sentence, embeddings[i][index], index)
                        else:
                            try:
                                ft_word = FetchedWord(word, tools.lemmatize(word), tag)
                                ft_word.update(sentence, embeddings[i][index], index)
                                fetched_rare_words.append(ft_word)
                            except:
                                print(embeddings.shape, index, sentence, tag)
        return webpage.count, fetched_rare_words, process_set(all_words)


    def similarity(word1, word2):
        emb1 = np.mean(word1.embeddings, axis=0)
        emb2 = np.mean(word2.embeddings, axis=0)
        return cosine_similarity(emb1.reshape(1, 1024), emb2.reshape(1, 1024))


    def relevance(s_words, f_words):
        matched_words = []
        matched_count = 0
        s_count = []
        f_count = []
        for word in f_words:
            index = find_word_root(s_words, word.root)
            if index >= 0:
                matched_words.append(word)
                if config.WITH_ELMO:
                    sim = similarity(word, s_words[index])[0][0]
                else:
                    sim = 1.0
                matched_count = matched_count + (0.6 * word.count + 1.0 * s_words[index].count) * sim
                if config.DEBUG and config.WITH_ELMO:
                    print("Similarity for the word: ", word.word, sim)
                if config.DEBUG and not config.WITH_ELMO:
                    print("Common word: ", word.word)
                s_count.append(s_words[index].count)
                f_count.append(word.count)
        return matched_words, matched_count


    def total(words):
        c = 0
        for word in words:
            c = c + word.count
        return c


    while True:
        url = input("Enter the URL or N\n")
        if url == "N":
            break
        print("Fetching the URL: ", url)
        web_count, fetched_rare_words, all_words = test_url(url)
        result, count = relevance(skill_rare_words, fetched_rare_words)
        meta = np.load(config.META_PATH)
        len_skills = len(config.SKILLS)
        meta = len_skills - len(meta)
        if config.DEBUG:
            print("META is", meta)
        score = (count / (total(fetched_rare_words))) * (21.0 / meta)
        if 0.4 < score < 0.6:
            print("Page vistied is moderately related: ", score)

        elif score >= 0.6:
            print("Page Visited is highly related: ", score)

        else:
            print("Page visited is not related: ", score)
