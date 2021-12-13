import config
from webpage import Webpage
from utils import tokenize
from utils import find_word
import utils
from fetched_word import FetchedWord
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

tools = utils.Utils()


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


# takes in a url and a custom elmo object and spits out words count, uncommon words list and all words list.
# Input:
# elmo - Custom Object
# URL  - webpage URL
# Output:
# webpage count  - number of words in the webpage ( Integer ).
# uncommon words - list of uncommon words in the webpage.
# all words      - list of all words present in the webpage.
def test_url(elmo, url):
    driver = config.DRIVER
    webpage = Webpage(url, driver)
    fetched = webpage.process_fetched()
    fetched_rare_words = []
    all_words = []
    print("Processing Text from the web page")
    start = time.time()
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
                        except Exception as e:
                            print(embeddings.shape, index, sentence, tag, e)
    if config.DEBUG:
        print("Time taken to run Elmo", time.time() - start, "for word count", len(all_words))
    return webpage.count, fetched_rare_words, process_set(all_words)


def similarity(word1, word2):
    emb1 = np.mean(word1.embeddings, axis=0)
    emb2 = np.mean(word2.embeddings, axis=0)
    return cosine_similarity(emb1.reshape(1, 1024), emb2.reshape(1, 1024))


# Loops through the fetched words and finds all the words that match with the extracted skill information.
# It then computes the context at which the words are used (similarity) using elmo and gives a score.
# All this score is added up and matched word count score is returned.
# Inputs:
# s_words - list of words that are uncommon in the knowledge base( skill information ).
# f_words - list of words that are uncommon in the extracted webpage( fetched words ).
# Outputs:
# matched_words       - List of words that are matching from knowledge base and the fetched webpage words.
# matched_count_score - each word that is matched is given a similarity score. This is computed based on the context
# at which this is used. This is given as a weight to calculate the score. These scores for all the words are combined
# to a sum score that is returned.
def relevance(s_words, f_words):
    matched_words = []
    matched_count_score = 0
    s_count = []
    f_count = []
    for word in f_words:
        # gives the index of the word in the extracted skill words list.
        index = utils.find_word_root(s_words, word.root)
        if index >= 0:
            matched_words.append(word)
            if config.WITH_ELMO:
                # Gives the similarity score of context for two same words used in webpages and in the skill
                # information.
                sim = similarity(word, s_words[index])[0][0]
            else:
                sim = 1.0
            # feature engineering, tried multiple combinations and decided to use these weights.
            matched_count_score = matched_count_score + (0.6 * word.count + 1.0 * s_words[index].count) * sim
            if config.DEBUG and config.WITH_ELMO:
                print("Similarity for the word: ", word.word, sim)
            if config.DEBUG and not config.WITH_ELMO:
                print("Common word: ", word.word)
            s_count.append(s_words[index].count)
            f_count.append(word.count)
    return matched_words, matched_count_score


def total(words):
    c = 0
    for word in words:
        c = c + word.count
    return c


# Takes in a list of matched scores sum returned by the relevance method and all the uncommon words present in the
# webpage as a list. It then computes score based on ratio of the matched count score and the length of the fetched
# words.
# Inputs
# matched_scores_sum - sum of all the matched words with any weights.
# fetched_rare_words - uncommon words present in the webpage that needs to be classified.
def calc_score(matched_scores_sum, fetched_rare_words):
    meta = np.load(config.META_PATH)
    len_skills = len(config.SKILLS)
    meta = len_skills - len(meta)
    if config.DEBUG:
        print("META is", meta)
    return (matched_scores_sum / (total(fetched_rare_words))) * (21.0 / meta)
