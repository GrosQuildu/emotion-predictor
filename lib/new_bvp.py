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
        measures = hb.process(self._y, self._freq)
        return measures['bpm']