import sys
import pickle
from lib.preprocessing import Preprocessing
from lib.postprocessing import Postprocessing
from lib.ai import AI
from os import listdir
from os.path import isfile, join
from sklearn import preprocessing


DATA_PATH = 'D:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'
DATA_FREQUENCY = 128
OUT_FILE = 'D:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\processed.dat'
NEED_PREPROCESSING = False


class Main:
    def loop(self):
        # while True:
        #     self._show_menu()
        #     option = input()
        #     self._handle_input(option)
        self._handle_input("1")

    def _handle_input(self, input):
        if input == "0":
            print("Finishing...")
            sys.exit(0)
        if input == "1":
            if NEED_PREPROCESSING:
                people = self._process_people(DATA_PATH)
                self._save_to_file(people)
            else:
                people = self._read_from_file()
            pp = Postprocessing(DATA_FREQUENCY)
            print("Postprocessing")
            x, y = pp.make_data_tuples(people)
            x_scaled = preprocessing.scale(x)
            print(x_scaled)
            ai = AI()
            ai.load_data(x_scaled, y)
            ai.test()


    def _show_menu(self):
        print("Choose an option:")
        print("1 - Preprocessing")
        print("0 - exit")

    def _process_people(self, directory):
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        preproc = Preprocessing()
        people = []

        print("Starting preprocessing")
        for file in files:
            person = preproc.process_person(f"{DATA_PATH}/{file}")
            print(f"{file} done. Got data from {len(person)} videos.")
            people.append(person)
            print(f"Done {file}")

        print("Preprocessing finished")
        return people

    def _save_to_file(self, data):
        with open(OUT_FILE, 'wb') as fp:
            pickle.dump(data, fp)

    def _read_from_file(self):
        with open(OUT_FILE, 'rb') as fp:
            return pickle.load(fp)


if __name__ == '__main__':
    main = Main()
    main.loop()
