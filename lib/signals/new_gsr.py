import matplotlib.pyplot as plt
import neurokit as nk
import numpy as np
import pickle
import pandas as pd
from statistics import mean, StatisticsError


class NewGSR:
    labels = [
        'avg_gsr',
        'tonic_avg',
        'phasic_avg',
        'peak_count',
        'amplitude_avg',
        'max_amplitude',
        'derivative_avg',
        'decrease_rate_avg',
        'derivative_negative_to_all',
        'local_minima_count',
        'local_maxima_count',
        'rising_time'
    ]

    def __init__(self, y, freq, file=False, filename=None):
        self._y = y
        self._freq = freq
        self._derivative = self._get_derivative(y)
        name = "neurokit/{}.dat".format(filename)
        if file:
            self.data = nk.eda_process(y, sampling_rate=freq)
            self.save_to_file(name, self.data)
        else:
            self.data = self.read_from_file(name)

    ################### NEUROKIT CACHE METHODS #####################

    def save_to_file(self, filename, data):
        with open(filename, 'wb') as fp:
            pickle.dump(data, fp)

    def read_from_file(self, filename):
        with open(filename, 'rb') as fp:
            return pickle.load(fp)

    ################################################################

    def get_features(self, extract_all_features=False):
        if extract_all_features:
            return self.get_all_features()
        return self.get_best_features()

    def get_best_features(self):
        return {}
        # return {
        #     'avg_gsr': self.avg()
        # }

    def get_all_features(self):
        return {
            'avg_gsr': self.avg(), #10
            'tonic_avg': self.tonic_avg(), #11
            'phasic_avg': self.phasic_avg(), #12
            'peak_count': self.peak_count(), #13
            'amplitude_avg': self.amplitude_avg(), #14
            'max_amplitude': self.max_aplitude(), #15
            'derivative_avg': self.derivative_avg(), #16
            'decrease_rate_avg': self.decrease_rate_avg(), #17
            'derivative_negative_to_all': self.derivative_negative_to_all(), #18
            'local_minima_count': self.local_minima_count(), #19
            'local_maxima_count': self.local_maxima_count(), #20
            'rising_time': self.rising_time() #21
        }

    def avg(self):
        return mean(self._y)

    def phasic_avg(self):
        return mean(self.data['df']['EDA_Phasic'])

    def tonic_avg(self):
        return mean(self.data['df']['EDA_Tonic'])

    def peak_count(self):
        return len(self.data['EDA']['SCR_Peaks_Indexes'])

    def amplitude_avg(self):
        return mean(self.data['EDA']['SCR_Peaks_Amplitudes'])

    def max_aplitude(self):
        return max(self.data['EDA']['SCR_Peaks_Amplitudes'])

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

    def _get_derivative(self, y):
        x = list(range(0, len(self._y)))
        dy = np.zeros(np.array(y).shape, np.float)
        dy[0:-1] = np.diff(y) / np.diff(x)

        return dy

    def plot(self, signal):
        frame = pd.DataFrame({"EDA": signal})
        frame.plot()
        plt.show()
