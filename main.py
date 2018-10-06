import sys
import pickle
import re
from lib.preprocessing import Preprocessing
from lib.postprocessing import Postprocessing
from lib.ai import AI
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


DATA_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'
ORIGINALS_PATH = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_original_bdf'
DATA_FREQUENCY = 128
OUT_FILE = 'F:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\processed.dat'
NEED_PREPROCESSING = True
EXTRACT_ALL_FEATURES = True


class Main:
    def loop(self):
        # while True:
        #     self._show_menu()
        #     option = input()
        #     self._handle_input(option)
        self._handle_input("3")

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

    def _train_model(self):
        x, y = self._get_data_tuples()
        ai = AI()
        ai.load_data(x, y)

        # print(x)
        # ai = AI()
        # ai.load_data(x, y)
        ai.test()

    def _sbs_scores(self):
        print("Looking for best features...")
        main.EXTRACT_ALL_FEATURES = True
        x, y = self._get_data_tuples()
        ai = AI()
        sbs = ai.sbs_score(x, y)

        k_feat = [len(k) for k in sbs.subsets_]
        plt.plot(k_feat, sbs.scores_, marker='o')
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.show()

        k5 = list(sbs.subsets_[8])
        print(k5)

    def _random_forest_scores(self):
        print("Looking for best features...")
        main.EXTRACT_ALL_FEATURES = True
        x, y = self._get_data_tuples()
        ai = AI()
        ai.random_forest_score(x, y, Preprocessing.get_labels())

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

    def _show_menu(self):
        print("Choose an option:")
        print("1 - Train model")
        print("2 - SBS")
        print("0 - exit")

    def _process_people(self, directory):
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        preproc = Preprocessing(DATA_FREQUENCY, extract_all_features=EXTRACT_ALL_FEATURES)
        people = []

        print("Starting preprocessing")
        for file in files:
            if file == "s13.dat":
                break

            number = self._get_file_number(file)
            person = preproc.process_person(
                f"{DATA_PATH}/{file}",
                f"{ORIGINALS_PATH}/s{number}.bdf",
                number
            )
            people.append(person)
            print(f"{file} done. Got data from {len(person)} videos.")

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
