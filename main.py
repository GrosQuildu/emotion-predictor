import sys
from lib.preprocessing import Preprocessing
from lib.postprocessing import Postprocessing
from lib.ai import AI
from os import listdir
from os.path import isfile, join


DATA_PATH = 'D:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'
DATA_FREQUENCY = 128


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
            people = self._process_people(DATA_PATH)
            pp = Postprocessing(DATA_FREQUENCY)
            print("Postprocessing")
            x, y = pp.make_data_tuples(people)
            ai = AI()
            ai.load_data(x, y)
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
            if file != "s08.dat":
                continue
            person = preproc.process_person(f"{DATA_PATH}/{file}")
            people.append(person)
            print(f"Done {file}")

        print("Preprocessing finished")
        return people

if __name__ == '__main__':
    main = Main()
    main.loop()
