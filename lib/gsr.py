import matplotlib.pyplot as plt
from statistics import mean, StatisticsError


class GSR:
    def __init__(self, y, timestamps, freq):
        self._y = y
        self._timestamps = timestamps
        self._freq = freq

    def match_timestamps(self, show_plot=False, avg=True):
        if show_plot:
            self._show_plot()

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
        plt.close('all')
        plt.figure(figsize=(32, 6))
        plt.plot(
            x,
            self._y
        )
        plt.show()
