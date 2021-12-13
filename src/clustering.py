import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import elmo
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import importlib
from utils import tokenize
from utils import Utils
from utils import find_word
from utils import find_word_root
import utils
from fetched_word import FetchedWord
import config
from webpage import Webpage
import inference_tools
import utils

if __name__ == '__main__':
    driver = config.DRIVER
    elmo = elmo.Elmo()
    tools = Utils()

    with open(config.SKILL_WORD_PATH, 'rb') as handle:
        skill_rare_words = pickle.load(handle)

    other_df = pd.read_excel("../data/web_history/other_website_history_xl.xls")
    skills_df = pd.read_excel("../data/web_history/synthetic_data_490_xl.xls")

    other_df = pd.DataFrame({"TIME": other_df['Visit Time'], "TITLE": other_df['Title'], "URL": other_df['URL']})

    urls = list(other_df['URL'])
    sample_fetched_words = []
    page_embedding_list = []
    for i, url in enumerate(tqdm(urls[:50])):
        print(i)
        _, fetched_words, _ = inference_tools.test_url(elmo, url)
        sample_fetched_words.append(fetched_words)
        emb = utils.get_web_embedding(fetched_words)
        try:
            assert emb.shape == (50, 1024) or emb.shape == (0,)
            page_embedding_list.append(emb)
        except Exception as e:
            print("Invalid embedding shape", e, "at url", url)

    np.save("../data/page_embeddings.npy", np.array(page_embedding_list))
