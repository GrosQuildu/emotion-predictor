import lib.emotion as em
from statistics import mean, StatisticsError
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import sys


DATA_BEGIN_SEC = 25
DATA_END_SEC = 45
WINDOW_SIZE_SEC = 5


class Postprocessing:
    def __init__(self, freq):
        self._freq = freq

    def make_data_tuples(self, people):
        x = []
        y = []

        for person in people:
            for image in person:
                bpm = self._get_avg_data(image['bpm'], DATA_BEGIN_SEC, DATA_END_SEC, WINDOW_SIZE_SEC)
                gsr = self._get_avg_data(image['gsr'], DATA_BEGIN_SEC, DATA_END_SEC, WINDOW_SIZE_SEC)

                emotion = em.get_class_for_values(image['valence'], image['arousal'])

                for i in range(0, len(bpm)-1):
                    x.append((bpm[i], gsr[i]))
                    y.append(emotion)

        return x, y

    def normalize(self, data):
        mms = MinMaxScaler()
        return mms.fit_transform(data)

    def standarize(self, data):
        stdsc = StandardScaler()
        return stdsc.fit_transform(data)

    def _get_avg_data(self, data, begin, end, window):
        avg_data = []
        values_to_mean = []
        next_target = (begin + window) * self._freq

        for i in data:
            # when loop is before begin
            if i[0] < self._freq * begin:
                continue

            # when loop reached end of the current window
            if i[0] > next_target:
                if len(values_to_mean) > 0:
                    avg_data.append(mean(values_to_mean))
                values_to_mean = []

                # current window was the last one, so we need to end the loop
                if next_target == end * self._freq:
                    break
                else:
                    next_target += window * self._freq

            # when loop is inside the window
            values_to_mean.append(i[1])

        return avg_data
