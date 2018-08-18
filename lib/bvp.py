import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, StatisticsError
import sys

DEFAULT_PLOT_SCOPE = 8064

# if difference between local max and local min is
# greater than this value, then the min is considered
# as a diastolic point
MIN_Y_DIFFERENCE = 600


class BVP:
    def __init__(self, x, y, freq):
        self._x = x
        self._y = y
        self._freq = freq

    def convert_to_bpm(self, show_plot=False, show_output_plot=False):
        dy = np.zeros(self._y.shape, np.float)
        dy[0:-1] = np.diff(self._y) / np.diff(self._x)
        extremes = self._get_local_extremes(self._x, dy)
        diastolic_points = self._get_diastolic_points(extremes)

        bpm = self._dialistic_points_to_beats(diastolic_points)

        if show_output_plot:
            self._show_plot([i[0]/self._freq for i in bpm], [i[1] for i in bpm])

        if show_plot:
            plt.close('all')
            plt.figure(figsize=(32, 6))
            plt.plot(
                self._x,
                dy,
                'b-',
                [i[0] for i in diastolic_points],
                [i[1] for i in diastolic_points],
                'ro'
            )
            plt.grid()
            plt.show()
            return []

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
            if y[i - 1] <= y[i] and y[i + 1] < y[i]:
                results.append((x[i], y[i], "max"))
                last = "max"

        return results

    def _get_diastolic_points(self, extremes):
        diastolic_points = []

        counting = True
        maxima = []
        for i in extremes:
            if i[2] == "min":
                continue
            if i[1] < 0:
                if counting:
                    lowest_min = self._get_lowest_min(maxima)
                    if lowest_min is not None:
                        diastolic_points.append(lowest_min)
                    maxima = []
                    counting = False

            else:
                counting = True
                if i[1] > 150:
                    maxima.append(i)

        return self._filter_low_diastolic_points(diastolic_points)
        #return diastolic_points

    # gets maximum that has the highest value
    def _get_lowest_min(self, maxima):
        highest = (0, -1)

        for i in maxima:
            if i[1] > highest[1]:
                highest = (i[0], i[1])

        if highest[1] > 50:
            return highest
        return None

    def _filter_low_diastolic_points(self, points):
        result = []
        # for the first point we take not the neighbors, but the 2th and 3th point
        avg = mean([points[1][1], points[2][1]])
        if points[0][1] > avg/2:
            result.append(points[0])

        # for ordinary points we check its neighbors
        for i in range(1, (len(points) - 1)):
            avg = mean([points[i-1][1], points[i+1][1]])
            if points[i][1] > avg/2:
                result.append(points[i])

        # for the last point we take the 2 points before it
        last = len(points) - 1
        avg = mean([points[last-2][1], points[last-1][1]])
        if points[last][1] > avg / 2:
            result.append(points[last])

        return result


    # converts list of diastolic points to beats per minute
    def _dialistic_points_to_beats(self, points):
        results = []
        for i in range(1, len(points)):
            try:
                bpm_value = self._interval_to_bpm(points[i][0] - points[i - 1][0])
                if 50 < bpm_value < 140:
                    results.append((round(points[i][0]), round(bpm_value, 2)))
            except TypeError:
                continue

        return results

    def _interval_to_bpm(self, interval):
        return (128 * 60) / interval
