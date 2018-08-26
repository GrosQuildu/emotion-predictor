import matplotlib.pyplot as plt
import sys
import numpy as np
from statistics import mean, StatisticsError


class GSR:
    def __init__(self, y, timestamps, freq):
        self._y = y
        self._timestamps = timestamps
        self._freq = freq

    def match_timestamps(self, show_plot=False, avg=True):
        if show_plot:
            self._show_plot()
            # sys.exit(0)

        results = []
        left = 0

        for i in self._timestamps:
            value = self._mean_values(left, i, self._y)
            left = i + 1
            results.append((i, value))

        if avg:
            results = self._values_to_diffs(results)
        return results

    def _mean_values(self, l, r, array):
        values_to_mean = array[l:r]
        return mean(values_to_mean)

    def _values_to_diffs(self, values):
        values_to_mean = [i[1] for i in values]
        avg = mean(values_to_mean)

        results = []
        for i in values:
            results.append((i[0], (i[1] - avg)))

        return results

    def _show_plot(self):
        x = list(range(0, len(self._y)))
        y = [i/20 for i in self._y]
        inv = self._convert(self._y)
        plt.close('all')
        plt.figure(figsize=(32, 6))
        plt.plot(
            x,
            y,
            'b-'
        )
        plt.grid()
        plt.show()

    def _convert(self, y):
        result = []
        for value in y:
            result.append(self._geneva_to_twente(value))
            # result.append(new_value)

        return result

    def _ohm_to_microsiemens(self, ohm):
        return (1 / ohm) * 1000.0

    def _geneva_to_twente(self, value):
        return (10**9)/value

    def _prepare(self, y):

        mini = min(y)
        result = []

        for value in y:
            result.append(value-mini)

        return result

    def _get_derivative(self, y):
        x = list(range(0, len(self._y)))
        dy = np.zeros(y.shape, np.float)
        dy[0:-1] = np.diff(y) / np.diff(x)

        return dy
