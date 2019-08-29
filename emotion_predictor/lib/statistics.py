from statistics import mean
from statistics import stdev

import matplotlib.pyplot as plt
import numpy as np


class Statistics:
    """
    This class is used to show corelation between physiological signal values and emption classes on graphs
    """
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels

    def create(self):
        avg_data = self._make_avg_data_dict()
        self._show_signal_graphs(avg_data)

    def _make_avg_data_dict(self):
        result = self._select_physical_values()
        return self._mean_physical_values(result)

    def _select_physical_values(self):
        result = {}

        for i in range(len(self.x)):
            emotion = self.y[i]
            if emotion not in result:
                result[emotion] = {}

            for j in range(len(self.x[i])):
                label = self.labels[j]
                if label not in result[emotion]:
                    result[emotion][label] = []

                result[emotion][label].append(self.x[i][j])

        return result

    def _mean_physical_values(self, values):
        for emotion in values:
            for signal in values[emotion]:
                tmp = values[emotion][signal]
                values[emotion][signal] = {}
                values[emotion][signal]['mean'] = mean(tmp)
                values[emotion][signal]['stdev'] = stdev(tmp)

        return values

    def _show_signal_graphs(self, values):
        labels = list(values.keys())

        signals = values[labels[0]].keys()
        for signal in signals:
            my_labels = list(labels)
            graph_data = []
            for i, emotion in enumerate(values):
                graph_data.append(values[emotion][signal]['mean'])
                my_labels[i] = my_labels[i] + " (" + str(round(values[emotion][signal]['stdev'], 2)) + ")"

            self._show_graph(signal, graph_data, my_labels)

    def _show_graph(self, title, data, labels):
        y_pos = np.arange(len(data))

        figure = plt.figure()
        figure.suptitle(title)
        plt.bar(y_pos, data, align='center', alpha=0.5)
        plt.xticks(y_pos, labels)

        plt.show()
