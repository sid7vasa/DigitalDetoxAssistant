DEBUG = False
LOG = True
SKILLS = ['Deep Learning', 'Convolutional Neural Networks', 'TensorFlow', 'Machine Learning', 'Algorithms',
          'Computer Vision', 'Deep Reinforcement Learning', 'Computer Science', 'Python', 'Matlab',
          'Recurrent Neural Networks', 'Django', 'Java', 'Keras', 'PyTorch', 'Neural Networks', 'Edge Computing',
          'OpenCV', 'Scikit-Learn', 'Natural Language Processing', 'Object Detection', 'Reinforcement Learning',
          'Generative Adversarial Networks']

# SKILLS = ['Unity Game Engine',
#           'Unreal Game Engine',
#           'Maya',
#           'Game Mechanics',
#           'Game Dynamics',
#           'Aesthetics',
#           'Gamification',
#           'Level design',
#           'Loot boxes',
#           'Intrinsic motivation',
#           'Oculus',
#           'Ray Tracing',
#           'Game Narrative',
#           'Haptic Technology',
#           'Role Playing Games',
#           'DLCs']



rare_word_threshold = 2700
DRIVER_PATH = r'D:\Downloads\chromedriver_win32\chromedriver.exe'
HEADLESS = True
SKILL_INFO_PATH = "../data/skills_elmo.csv"
WITH_ELMO = True
if WITH_ELMO:
    SKILL_WORD_PATH = '../data/skill_words.pkl'
else:
    SKILL_WORD_PATH = '../data/skill_words_without_elmo.pkl'
TEST_URL = "https://www.wired.com/story/should-anyone-actually-care-about-ray-tracing/"
ELMO_PATH = "../word_representation_model/elmo"
META_PATH = "../data/meta.npy"

