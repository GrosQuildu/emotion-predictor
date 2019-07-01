import pickle
import re
import sys
import warnings
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

from config import DATA_FREQUENCY, NEED_PREPROCESSING, PICKLED_DATA_RESTING, PICKLED_DATA_EMOTIONIZED, \
    PICKLED_DATA_PICTURES, OUT_FILE, INITIAL_ESTIMATORS, \
    OPTIMIZED_ESTIMATORS
from lib.accuracy.cross_validation import MultimodelCrossValidator
from lib.ai import AI
from lib.postprocessing import Postprocessing
from lib.preprocessing import Preprocessing
from lib.statistics import Statistics
from lib.predictor import Predictor

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class Main:
    """
    This is an endpoint class that interacts with user and triggers all actions
    """
    def loop(self):
        """
        Method containing main loop
        """
        while True:
            self._show_menu()
            option = input()
            self._handle_input(option)

    def _handle_input(self, value):
        if value == "0":
            print("Finishing...")
            sys.exit(0)
        if value == "1":
            self._train_model()
        elif value == "2":
            self._sbs_scores()
        elif value == "3":
            self._random_forest_scores()
        elif value == "4":
            self._reverse_sbs_scores()
        elif value == "5":
            self._analyse_predictions()
        elif value == "6":
            self._show_statistics()
        elif value == "7":
            self._validate_estimators()
        elif value == "8":
            self._optimize_best_estimators()
        elif value == "9":
            self._validate_optimized_estimators()
        elif value == "10":
            self._analyze_classes()
        elif value == "11":
            self._run_predicting()
        else:
            print("Incorrect input")

    ###########################################################################
    # METHODS THAT HANDLE INPUT                                               #
    ###########################################################################

    def _run_predicting(self):
        x, y = self._get_data_tuples()

        print("Enter signal frequency (Hz): ")
        frequency = input()

        ai = AI()
        model = ai.get_best_model(x, y)
        predictor = Predictor(frequency, model)

    def _train_model(self):
        print("Training model")
        x, y = self._get_data_tuples()
        ai = AI()

        ai.simple_test(x, y)

    def _sbs_scores(self):
        print("Looking for best features...")
        x, y = self._get_data_tuples()
        ai = AI()
        sbs = ai.sbs_score(x, y)

        k_feat = [len(k) for k in sbs.subsets_]
        plt.plot(k_feat, sbs.scores_, marker='o')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.show()

        for subset in sbs.subsets_:
            if len(subset) == 4:
                k4 = list(subset)
                print(k4)
                break

    def _reverse_sbs_scores(self):
        print("Looking for best features...")
        x, y = self._get_data_tuples()
        print(len(y))
        ai = AI()
        features, accuracy = ai.reverse_sbs_score(x, y)

        labels = Preprocessing.get_labels()
        print(f"Max accuracy is {accuracy}")
        print("Features:")

        for f in features:
            print(labels[f])

    def _random_forest_scores(self):
        print("Looking for best features...")
        x, y = self._get_data_tuples()
        ai = AI()
        ai.random_forest_score(x, y, Preprocessing.get_labels())

    def _analyse_predictions(self):
        print("Analysing predictions")
        x, y = self._get_data_tuples()
        ai = AI()
        main, detail = ai.analyse_predictions(x, y)
        print(f"Main accuracy = {main}")
        print(detail)

    def _show_statistics(self):
        print("Creating statistics...")
        main.EXTRACT_ALL_FEATURES = True
        x, y = self._get_data_tuples()
        labels = Preprocessing.get_labels()
        # print(y)
        stats = Statistics(x, y, labels)
        stats.create()

    def _validate_estimators(self):
        x, y = self._get_data_tuples()
        print("Validating models with default parameters")

        validator = MultimodelCrossValidator(x, y, INITIAL_ESTIMATORS)
        results = validator.validate_all()
        results.sort(key=lambda el: el[0], reverse=True)

        print("Final scores:")
        for mean, name in results:
            print(f"{name}: {mean}")

    def _optimize_best_estimators(self):
        x, y = self._get_data_tuples()
        print("Optimizing model")

        ai = AI()
        score, params = ai.optimize_best_estimators(x, y)
        print(score, params)

    def _validate_optimized_estimators(self):
        x, y = self._get_data_tuples()
        print("Validating models with optimized parameters")

        validator = MultimodelCrossValidator(x, y, OPTIMIZED_ESTIMATORS)
        results = validator.validate_all()
        results.sort(key=lambda el: el[0], reverse=True)

        print("Final scores:")
        for mean, name in results:
            print(f"{name}: {mean}")

    def _analyze_classes(self):
        """
        Calculates how much samples are in particular classes
        """
        x, y = self._get_data_tuples()

        class_count = {}
        for cl in y:
            if cl not in class_count:
                class_count[cl] = 0

            class_count[cl] += 1

        all = len(y)
        for cl in class_count:
            class_count[cl] = class_count[cl]/all*100

        print(class_count)

    ###########################################################################
    # OTHER PRIVATE METHODS                                                   #
    ###########################################################################

    def _show_menu(self):
        print("Choose an option:")
        print("1 - Train model")                    # Basic accuracy test using RandomForestClassifier
        print("2 - SBS")                            # select features
        print("3 - Random Forest scores")           # select features
        print("4 - Reverse SBS scores")             # select features
        print("5 - Analyze preditctions")           # <- there was a bug here, idk
        print("6 - Show statistics")                # some plots, idk
        print("7 - Validate estimators")            # chose best models
        print("8 - Optimize best estimators")       # select best models' parameters
        print("9 - Validate optimized estimators")  # test models
        print("10 - Show class statistics")         # idk
        print("11 - Predict emotions on live data") #
        print("0 - exit")

    def _get_data_tuples(self):
        if NEED_PREPROCESSING:
            people = self._process_people(DATA_PATH)
            self._save_to_file(people)
        else:
            people = self._read_from_file()
        pp = Postprocessing(DATA_FREQUENCY)
        print("Postprocessing")
        x, y = pp.make_data_tuples(people)
        x_scaled = pp.standarize(x)

        return x_scaled, y

    def _process_people(self, directory):
        preproc = Preprocessing(DATA_FREQUENCY)
        people = []
        num_samples = 0

        print("Starting preprocessing")
        for person in preproc.process_person(PICKLED_DATA_EMOTIONIZED, PICKLED_DATA_RESTING, PICKLED_DATA_PICTURES):
            try:
                people.append(person)
                person_sample_count = len(person)
                print(f"next done. Got data from {person_sample_count} images.")
                num_samples += person_sample_count
                print(f"Has {num_samples} samples already")
            except Exception:
                # raise
                pass

        print("Preprocessing finished")
        NEED_PREPROCESSING = False  # raz powinno wystarczyc afaik
        return people

    def _save_to_file(self, data):
        with open(OUT_FILE, 'wb') as fp:
            pickle.dump(data, fp)

    def _read_from_file(self):
        with open(OUT_FILE, 'rb') as fp:
            return pickle.load(fp)

    def _get_file_number(self, name):
        m = re.search("s([0-9]{2})\.", name)
        return m.group(1)


if __name__ == '__main__':
    main = Main()
    main.loop()
