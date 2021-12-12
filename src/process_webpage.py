from elmo import Elmo
from fetched_word import FetchedWord
from webpage import Webpage
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from utils import tokenize
from utils import Utils
from utils import find_word


if __name__ == '__main__':
    options = Options()
    options.add_argument('headless')
    driver = webdriver.Chrome(executable_path=r'D:\Downloads\chromedriver_win32\chromedriver.exe', options=options)
    webpage = Webpage("https://machinelearningmastery.com/what-is-deep-learning/", driver)
    fetched = webpage.process_fetched()
    elmo = Elmo()
    tools = Utils()

    fetched_rare_words = []
    for tag in fetched.keys():
        if len(fetched[tag]) == 0:
            continue
        try:
            embeddings = elmo.run(fetched[tag])
        except:
            print(fetched, tag)
        for i, sentence in enumerate(fetched[tag]):
            words = tokenize(sentence)
            for word in words:
                if tools.is_rare(word):
                    index = words.index(word)
                    print("uncommon: ", word)
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