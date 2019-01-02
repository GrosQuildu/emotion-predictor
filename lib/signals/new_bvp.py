import numpy as np
import lib.signals.heartbeat as hb
from biosppy.signals.bvp import bvp
from collections import OrderedDict


class NewBVP:
    labels = [
        'bpm',      #0
        'pnn20',    #1
        'pnn50',    #2
        'hr_mad',   #3
        'rmssd',    #4
        'sdsd',     #5
        'sdnn',     #6
        'lf',       #7
        'hf',       #8
        'lf/hf'     #9
    ]

    def __init__(self, signal, freq, show_plot=False):
        self.plot = show_plot
        self.measures = self._process_signal(signal, freq)
        if self.measures['bpm'] > 125 or self.measures['bpm'] < 50:
            raise Exception(f"Malformed data: BPM={self.measures['bpm']}")

    def get_features(self, extract_all_features=False):
        if extract_all_features:
            return self.get_all_features()
        return self.get_best_features()

    def get_best_features(self):
        return OrderedDict([
            ('bpm', self.measures['bpm']),
            ('pnn20', self.measures['pnn20'])
        ])

    def get_all_features(self):
        result = OrderedDict()
        for label in self.labels:
            result[label] = self.measures[label]

        return result

    def _process_signal(self, signal, freq):
        biosppy_processed = bvp(signal, freq, show=self.plot)
        y_filtered = biosppy_processed['filtered']
        if isinstance(y_filtered, list):
            y = np.asarray(y_filtered)
        else:
            y = y_filtered

        measures = hb.process(y, freq, calc_freq=True)
        return measures
