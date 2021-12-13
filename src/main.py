import pickle

import inference_tools
from utils import Utils
import config

if config.WITH_ELMO:
    import elmo
else:
    import dummyElmo

if __name__ == '__main__':
    driver = config.DRIVER
    if config.WITH_ELMO:
        elmo = elmo.Elmo()
    else:
        elmo = dummyElmo.Elmo()

    if config.DEBUG: print("loading skill words from: ", config.SKILL_WORD_PATH)
    tools = Utils()
    print("Opening your skill information: ")
    with open(config.SKILL_WORD_PATH, 'rb') as handle:
        skill_rare_words = pickle.load(handle)

    while True:
        url = input("Enter the URL or N:\n")
        if url == "N":
            break
        print("Fetching the URL: ", url)
        web_count, fetched_rare_words, all_words = inference_tools.test_url(elmo, url)
        matched_words, matched_scores_sum = inference_tools.relevance(skill_rare_words, fetched_rare_words)
        score = inference_tools.calc_score(matched_scores_sum, fetched_rare_words)

        if 0.4 < score < 0.6:
            print("Page visited is moderately related: ", score)

        elif score >= 0.6:
            print("Page visited is highly related: ", score)

        else:
            print("Page visited is not related: ", score)
