import pandas as pd
import numpy as np
import config
import elmo
import dummyElmo
import utils
from utils import tokenize
from utils import find_word
from utils import preprocess_sentence_text
from skill_word import SkillWord
import pickle
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import wikipedia as wiki



if __name__ == '__main__':
    skills = config.SKILLS
    options = Options()
    options.headless = config.HEADLESS
    options.add_argument("--window-size=1920,1200")
    driver = webdriver.Chrome(executable_path=config.DRIVER_PATH, options=options)
    skillset = pd.Series(skills)

    skill_summary_list = []
    li = []
    print("Extracting Skill Knowledge Base: ", flush=True)
    for i, skill in enumerate(tqdm(skillset)):
        skill = skill.lower()
        try:
            skill_summary = wiki.WikipediaPage(skill).summary
            skill_summary_list.append(skill_summary)
        except Exception as e:
            li.append(skill)
            skill_summary_list.append("")
    if len(li) > 0:
        print("Couldn't find any information on: ", li)
    df = pd.DataFrame({"skills": skills, "summary": skill_summary_list})
    print("Dumping to :", config.SKILL_INFO_PATH)
    df.to_csv(config.SKILL_INFO_PATH)
    np.save(config.META_PATH, np.array(li))

    print("Information Retrieved.")
    print("Loading Elmo for Deep Contextualized Word Representation")
    if config.WITH_ELMO:
        elmo = elmo.Elmo()
    else:
        elmo = dummyElmo.Elmo()
    print("Done")

    tools = utils.Utils()
    print(config.SKILL_INFO_PATH, "...")
    skills_df = pd.read_csv(config.SKILL_INFO_PATH)
    skills_df = skills_df.drop(columns=["Unnamed: 0"])

    skill_rare_words = []
    rare_words = []
    skill_words_count = 0
    for skill_ind, skill in enumerate(tqdm(skills_df['summary'])):
        text = skills_df['summary'][skill_ind]
        skill_name = skills_df['skills'][skill_ind]
        if isinstance(text, float) and np.isnan(text):
            continue
        sentences = preprocess_sentence_text(text)
        embeddings = elmo.run(sentences)
        for i, sentence in enumerate(sentences):
            words = tokenize(sentence)
            skill_words_count = skill_words_count + len(words)
            for word in words:
                if tools.is_rare(word):
                    index = words.index(word)
                    word_ind = find_word(skill_rare_words, word)
                    try:
                        if word_ind >= 0:
                            skill_rare_words[word_ind].update(sentence, embeddings[i][index], index, skill_name)
                        else:
                            sk_word = SkillWord(word, tools.lemmatize(word))
                            sk_word.update(sentence, embeddings[i][index], index, skill_name)
                            skill_rare_words.append(sk_word)
                    except Exception as e:
                        print("EXCEPTION! AT EXTRACTING SKILL INFORMATION", word, sentence, i)
                        print(e)
                        continue
    with open(config.SKILL_WORD_PATH, 'wb') as handle:
        pickle.dump(skill_rare_words, handle)
        print("Dumped extracted words at: ", handle.name)
