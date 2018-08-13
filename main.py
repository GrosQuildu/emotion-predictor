import sys
from lib.preprocessing import Preprocessing


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
            preproc = Preprocessing()
            preproc.process_person('D:\dane inz\DEAP (Database for Emotion Analysis using Physiological Signals)\data_preprocessed_python\s01.dat')

    def _show_menu(self):
        print("Choose an option:")
        print("1 - Preprocessing")
        print("0 - exit")


if __name__ == '__main__':
    main = Main()
    main.loop()
