import numpy as np
import lib.heartbeat as hb


class NewBVP:
    def __init__(self, x, y, freq):
        self._x = x
        if isinstance(y, list):
            self._y = np.asarray(y)
        else:
            self._y = y
        self._freq = freq

    def convert_to_bpm(self, show_plot=False, show_output_plot=False):
        measures = hb.process(self._y, self._freq, calc_freq=True)
        print(measures)
        # return measures['bpm'], measures['hr_mad'] # 0.21
        # return measures['bpm'], measures['pnn50'] # 0.26
        return measures['bpm'], measures['pnn20'] # 0.27
        # return measures['bpm'], measures['rmssd'] # 0.22
        # return measures['bpm'], measures['sdsd'] # 0.23
        # return measures['bpm'], measures['sdnn'] # 0.24
        # return measures['bpm'], measures['lf'] # 0.24
        # return measures['bpm'], measures['hf'] # 0.24
        # return measures['bpm'], measures['lf/hf'] # 0.23
