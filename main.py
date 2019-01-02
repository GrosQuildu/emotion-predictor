import sys
import pickle
import re
from lib.preprocessing import Preprocessing
from lib.postprocessing import Postprocessing
from lib.ai import AI
from lib.statistics import Statistics
from lib.accuracy.cross_validation import MultimodelCrossValidator
from os import listdir
from os.path import isfile, join
from config import DATA_FREQUENCY, NEED_PREPROCESSING, DATA_PATH, ORIGINALS_PATH, OUT_FILE, INITIAL_ESTIMATORS
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class Main:
    def loop(self):
        # while True:
        #     self._show_menu()
        #     option = input()
        #     self._handle_input(option)
        self._handle_input("7")

    def _handle_input(self, input):
        if input == "0":
            print("Finishing...")
            sys.exit(0)
        if input == "1":
            self._train_model()
        elif input == "2":
            self._sbs_scores()
        elif input == "3":
            self._random_forest_scores()
        elif input == "4":
            self._reverse_sbs_scores()
        elif input == "5":
            self._analyse_predictions()
        elif input == "6":
            self._show_statistics()
        elif input == "7":
            self._validate_estimators()

    ###########################################################################
    # METHODS THAT HANDLE INPUT                                               #
    ###########################################################################

    def _train_model(self):
        print("Training model")
        x, y = self._get_data_tuples()
        ai = AI()
        ai.load_data(x, y)

        ai.test()

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
        ai.load_data(x, y)
        main, detail = ai.split_predictions()
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
        print("Validating estimators with default parameters")

        validator = MultimodelCrossValidator(x, y, INITIAL_ESTIMATORS)
        results = validator.validate_all()
        results.sort(key=lambda el: el[0], reverse=True)

        print("Final scores:")
        for mean, name in results:
            print(f"{name}: {mean}")

    ###########################################################################
    # OTHER PRIVATE METHODS                                                   #
    ###########################################################################

    def _show_menu(self):
        print("Choose an option:")
        print("1 - Train model")
        print("2 - SBS")
        print("3 - Random Forest scores")
        print("4 - Reverse SBS scores")
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
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        preproc = Preprocessing(DATA_FREQUENCY)
        people = []
        num_samples = 0

        print("Starting preprocessing")
        for file in files:
            # if file == "s23.dat":
            #     break

            try:
                number = self._get_file_number(file)
                person = preproc.process_person(
                    f"{DATA_PATH}/{file}",
                    f"{ORIGINALS_PATH}/s{number}.bdf",
                    number
                )
                people.append(person)
                person_sample_count = len(person)
                print(f"{file} done. Got data from {person_sample_count} videos.")
                num_samples += person_sample_count
                print(f"Has {num_samples} already")
            except Exception:
                # raise
                pass

        print("Preprocessing finished")
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
