import numpy as np
import lib.signals.heartbeat as hb


class NewBVP:
    labels = [
        'bpm', #0
        'pnn20', #1
        'pnn50', #2
        'hr_mad', #3
        'rmssd', #4
        'sdsd', #5
        'sdnn', #6
        'lf', #7
        'hf', #8
        'lf/hf' #9
    ]

    def __init__(self, x, y, freq):
        self._x = x
        if isinstance(y, list):
            self._y = np.asarray(y)
        else:
            self._y = y
        self._freq = freq
        self.measures = hb.process(self._y, self._freq, calc_freq=True)
        if self.measures['bpm'] > 125 or self.measures['bpm'] < 50:
            raise Exception("Malformed data")

    def get_features(self, extract_all_features=False):
        if extract_all_features:
            return self.get_all_features()
        return self.get_best_features()

    def get_best_features(self):
        # return {
        #     'bpm': self.measures['bpm'],
        #     'pnn20': self.measures['pnn20'],
        #     'pnn50': self.measures['pnn50'],
        # }
        return {
            'bpm': self.measures['bpm']
        }

    def get_all_features(self):
        result = {}
        for label in self.labels:
            result[label] = self.measures[label]

        return result
