import sys
from lib.preprocessing import Preprocessing
from os import listdir
from os.path import isfile, join


DATA_PATH = 'D:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python'


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
            self._process_people(DATA_PATH)

    def _show_menu(self):
        print("Choose an option:")
        print("1 - Preprocessing")
        print("0 - exit")

    def _process_people(self, directory):
        files = [f for f in listdir(directory) if isfile(join(directory, f))]
        preproc = Preprocessing()
        count = 0

        for file in files:
            preproc.process_person(f"{DATA_PATH}/{file}")
            count += 1
            if count >= 1:
                break


if __name__ == '__main__':
    main = Main()
    main.loop()
