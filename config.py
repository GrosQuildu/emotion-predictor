from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# PATH
DATA_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'
ORIGINALS_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_original_bdf'
OUT_FILE = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\processed.dat'

# FREQUENCY
DATA_FREQUENCY = 128

# SWITCHES
NEED_PREPROCESSING = True
EXTRACT_ALL_FEATURES = False

# SIGNAL TRIMMING
SIGNAL_BEGIN = 20
SIGNAL_END = 60

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

# MISCELLANEOUS
CPU_CORES_NUM = 2
