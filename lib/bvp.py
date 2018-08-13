import matplotlib.pyplot as plt
from statistics import mean, StatisticsError

DEFAULT_PLOT_SCOPE = 8064


class BVP:
    def __init__(self, x, y, freq):
        self._x = x
        self._y = y
        self._freq = freq

    def convert_to_bpm(self, show_plot=False, avg=True):
        extremes = self._get_local_extremes(self._x, self._y)
        diastolic_points = self._get_diastolic_points(extremes)
        bpm = self._dialistic_points_to_beats(diastolic_points)

        if show_plot:
            plt.close('all')
            plt.figure(figsize=(32, 6))
            plt.plot(
                self._x,
                self._y,
                'b-',
                [i[0] for i in diastolic_points],
                [i[1] for i in diastolic_points],
                'ro'
            )
            plt.show()

        if avg:
            bpm = self._values_to_diffs(bpm)

        return bpm

    def plot_original(self, scope=DEFAULT_PLOT_SCOPE):
        self._show_plot(self._x[:scope], self._y[:scope])

    def _show_plot(self, x, y, type='-'):
        plt.plot(x, y, type)
        plt.show()

    def _get_local_extremes(self, x, y):
        if len(x) != len(y):
            raise ValueError("Length of X and Y must be equal")

        results = []

        for i in range(1, (len(x) - 1)):
            # check for local minimum
            if y[i - 1] >= y[i] and y[i + 1] > y[i]:
                results.append((x[i], y[i], "min"))
                last = "min"
            if y[i - 1] < y[i] and y[i + 1] < y[i]:
                results.append((x[i], y[i], "max"))
                last = "max"

        return results

    def _get_diastolic_points(self, extremes):
        diastolic_points = []
        last_x = None

        for i in extremes:
            if i[2] == "min":
                if (last_x is not None) and (i[0] - last_x) > 50:
                    diastolic_points.append(i)

            last_x = i[0]

        return diastolic_points

    # gets minimum that has the lowest value
    def _get_lowest_min(self, minima):
        lowest = (0, 999)

        for i in minima:
            if i[1] < lowest[1]:
                lowest = (i[0], i[1])

        if lowest[1] < 0:
            return lowest
        return None

    # converts list of diastolic points to beats per minute
    def _dialistic_points_to_beats(self, points):
        results = []
        for i in range(1, len(points)):
            try:
                bpm_value = self._interval_to_bpm(points[i][0] - points[i - 1][0])
                #if 50 < bpm_value < 130:
                results.append((round(points[i][0] * 1000), round(bpm_value, 2)))
            except TypeError:
                continue

        return results

    def _interval_to_bpm(self, interval):
        return (128 * 60) / interval

    def _values_to_diffs(self, values):
        originals = []
        for i in values:
            originals.append(i[1])

        avg = mean(originals)
        results = []

        for i in values:
            results.append((i[0], round(i[1] - avg, 2)))

        return results
