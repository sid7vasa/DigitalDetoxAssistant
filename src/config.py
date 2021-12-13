from selenium import webdriver
from selenium.webdriver.chrome.options import Options

DEBUG = False
LOG = True
SKILLS_CS = ['Deep Learning', 'Convolutional Neural Networks', 'TensorFlow', 'Machine Learning', 'Algorithms',
             'Computer Vision', 'Deep Reinforcement Learning', 'Computer Science', 'Python', 'Matlab',
             'Recurrent Neural Networks', 'Django', 'Java', 'Keras', 'PyTorch', 'Neural Networks', 'Edge Computing',
             'OpenCV', 'Scikit-Learn', 'Natural Language Processing', 'Object Detection', 'Reinforcement Learning',
             'Generative Adversarial Networks']

SKILLS_GM = ['Unity Game Engine', 'Unreal Game Engine', 'Maya', 'Game Mechanics', 'Game Dynamics', 'Aesthetics',
             'Gamification', 'Level design', 'Loot boxes', 'Intrinsic motivation', 'Oculus', 'Ray Tracing',
             'Game Narrative', 'Haptic Technology', 'Role Playing Games', 'DLCs']

SKILLS = SKILLS_CS

# Elmo model location for tensorflow hub to load and execute.
ELMO_PATH = "../word_representation_model/elmo"

# Bert Rare Word Threshold.
rare_word_threshold = 2700

# Selenium Config
HEADLESS = True
DRIVER_PATH = r"D:\Downloads\chromedriver_win32\chromedriver.exe"
options = Options()
options.headless = HEADLESS
options.add_argument("--window-size=1920,1200")
DRIVER = webdriver.Chrome(executable_path=DRIVER_PATH, options=options)

# Skills summaries extracted information saved file.
SKILL_INFO_PATH = "../data/skills_elmo.csv"

# Rare word skills with embeddings.
WITH_ELMO = False
if WITH_ELMO:
    SKILL_WORD_PATH = '../data/skill_words.pkl'
else:
    SKILL_WORD_PATH = '../data/skill_words_without_elmo.pkl'

# Test url if needed
TEST_URL = "https://www.geeksforgeeks.org/binary-tree-set-1-introduction/"

# Meta is used to factor the number of skills given by the user.
# This is used to balance the score based on the number of skill keywords extracted.
META_PATH = "../data/meta.npy"
