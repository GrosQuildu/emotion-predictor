import matplotlib.pyplot as plt
import sys
import numpy as np
import copy
import imp
from array import array
from statistics import mean, StatisticsError
from biosppy.signals import eda
from pprint import pprint


class GSR:
    def __init__(self, y, freq):
        self._y = copy.deepcopy(y)
        self._freq = freq
        self._derivative = self._get_derivative(copy.deepcopy(y))
        self.data = eda.eda(signal=copy.deepcopy(y), sampling_rate=copy.deepcopy(freq), show=False)

    #### METHODS ADDED IN EXPERIMENTAL VERSION ###########

    def convert(self):
        return {
            'avg_gsr': self.avg()
            # 'peak_count': self.peek_count()
        }

    def avg(self):
        return mean(self._y)

    def derivative_avg(self):
        return mean(self._derivative)

    def decrease_rate_avg(self):
        values = []
        for value in self._derivative:
            if value < 0:
                values.append(value)

        return mean(values)

    def derivative_negative_to_all(self):
        n = 0
        for value in self._derivative:
            if value < 0:
                n += 1

        return n/len(self._derivative)

    def local_minima_count(self):
        n = 0
        for i in range(1, len(self._derivative)-1):
            if self._derivative[i-1] > self._derivative[i] and \
            self._derivative[i+1] > self._derivative[i]:
                n += 1

        return n

    def local_maxima_count(self):
        n = 0
        for i in range(1, len(self._derivative) - 1):
            if self._derivative[i - 1] < self._derivative[i] and \
                    self._derivative[i + 1] < self._derivative[i]:
                n += 1

        return n

    def rising_time(self):
        n = 0
        for value in self._derivative:
            if value > 0:
                n += 1

        return n / self._freq

    def peek_count(self):
        self._process_eda_signal()

        return len(self.data['peaks'])

    def peek_avg(self):
        self._process_eda_signal()

        val = 0
        for peak in self.data['peaks']:
            val += peak

        try:
            return val / len(self.data['peaks'])
        except ZeroDivisionError:
            return 0

    def ampl_avg(self):
        self._process_eda_signal()

        try:
            return mean(self.data['amplitudes'])
        except StatisticsError:
            return 0

    def _process_eda_signal(self):
        if not self.data:
            self.data = eda.kbk_scr(signal=copy.deepcopy(self._y), sampling_rate=self._freq)

    #### METHODS FROM ORIGINAL VERSION ###################

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
        dy = np.zeros(np.array(y).shape, np.float)
        dy[0:-1] = np.diff(y) / np.diff(x)

        return dy
