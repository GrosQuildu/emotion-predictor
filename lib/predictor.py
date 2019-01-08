import sys

from lib.postprocessing import Postprocessing
from lib.preprocessing import Preprocessing


class Predictor:
    def __init__(self, freq, model):
        self.freq = freq
        self.model = model

        self.pre = Preprocessing(self.freq)
        self.post = Postprocessing(self.freq)

        self.base_bvp_features = None
        self.base_gsr_features = None

    def process(self):
        base_bvp, base_gsr = self._get_basic_values()

        print("Processing base signals...")
        self.base_bvp_features = self.pre.get_base_bvp_features(base_bvp)
        self.base_gsr_features = self.pre.get_base_gsr_features(base_gsr)

        while True:
            print("Enter BVP signal (one value per line). Press ENTER to finish: ")
            bvp = self._read_signal()
            print("Enter GSR signal (one value per line). Press ENTER to finish: ")
            gsr = self._read_signal()

            values = self.pre.get_diffed_values(bvp, gsr, self.base_bvp_features, self.base_gsr_features)
            values = self.post.standarize(values)

            emotion = self.model.predict(values)
            print(f"Predicted emotion class: {emotion}")

    def _read_signal(self):
        signal = []

        for line in sys.stdin:
            rawline = line.rstrip()
            if not rawline:
                break
            signal.append(float(rawline))

        return signal

    def _get_basic_values(self):
        print("Enter base BVP signal (one value per line). Press ENTER to finish: ")
        base_bvp = self._read_signal()
        print("Enter base GSR signal (one value per line). Press ENTER to finish: ")
        base_gsr = self._read_signal()

        return base_bvp, base_gsr

