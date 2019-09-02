from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from os.path import join as pj

# old/original config
# DATA_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'
# ORIGINALS_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_original_bdf'
# OUT_FILE = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\processed.dat'


# base dir for all the data and output files
BASE_DIR_DATA = '/home/gros/studia/eaiib_5/wshop/data'

# files from experiments
PATH_DATA = '2018-afcai-spring'
PATH_PICTURES = 'NAPS_valence_arousal_2014.csv'

# may be any writable path that EXISTS and arbitrary filename
PICKLED_DATA_RESTING = 'geist_preproc/preprocessed_geist_resting.pickle'
PICKLED_DATA_EMOTIONIZED = 'geist_preproc/preprocessed_geist_emotionized.pickle'
PICKLED_DATA_PICTURES = 'geist_preproc/preprocessed_geist_pictures.pickle'
PICKLED_PREPROCESSED = 'geist_preproc/preprocessed_data.pickle'

# any writable directory that EXISTS
NEUROKIT_PATH = 'neurokit'

# ---------- make absolute paths
PATH_DATA = pj(BASE_DIR_DATA, PATH_DATA)
PATH_PICTURES = pj(BASE_DIR_DATA, PATH_PICTURES)
PICKLED_DATA_RESTING = pj(BASE_DIR_DATA, PICKLED_DATA_RESTING)
PICKLED_DATA_EMOTIONIZED = pj(BASE_DIR_DATA, PICKLED_DATA_EMOTIONIZED)
PICKLED_DATA_PICTURES = pj(BASE_DIR_DATA, PICKLED_DATA_PICTURES)
NEUROKIT_PATH = pj(BASE_DIR_DATA, NEUROKIT_PATH)
OUT_FILE = pj(BASE_DIR_DATA, PICKLED_PREPROCESSED)
# ---------- 

DO_LOGS = 1

# FREQUENCY
DATA_FREQUENCY = 128

# SWITCHES
NEED_PREPROCESSING = True
EXTRACT_ALL_FEATURES = False
SHOW_PLOTS = False

# SIGNAL TRIMMING
SIGNAL_BEGIN = 1
SIGNAL_END = 8

# MACHINE LEARNING
VALIDATION_SIZE = 0.2
SEED = 1

# INITIAL ESTIMATORS TO VALIDATE
INITIAL_ESTIMATORS = [
    MLPClassifier,
    KNeighborsClassifier,
    SVC,
    GaussianProcessClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GaussianNB,
    QuadraticDiscriminantAnalysis
]

OPTIMIZED_ESTIMATORS = [
    (SVC, {
        'random_state': 1,
        'kernel': 'linear',
        'C': 0.1,
        'probability': True,
        'decision_function_shape': 'ovo'
    }),
    (MLPClassifier, {
        'random_state': 1,
        'activation': 'identity',
        'solver': 'lbfgs',
        'hidden_layer_sizes': (100, 100),
        'alpha': 0
    }),
    (GaussianProcessClassifier, {
        'random_state': 1,
        'optimizer': None,
        'max_iter_predict': 2,
        'multi_class': 'one_vs_rest'
    }),
    (DecisionTreeClassifier, {
        'random_state': 1,
        'criterion': 'gini',
        'max_depth': 7,
        'splitter': 'random',
        'presort': True
    })
]

# MISCELLANEOUS
CPU_CORES_NUM = 2

# possible values are: valence, arousal, both
CLASSIFICATION_METHOD = "both"
