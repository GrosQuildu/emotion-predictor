import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from lib import emotion
import sys

class Analysis:
    def __init__(self, data):
        self.data = self._group_by_emotion(data)

    def make_data_plots(self):
        labels = [i for i in self.data]
        n = len(labels)
        x = np.arange(n)
        values_bpm = [self.data[key]['bpm'] for key in self.data]
        values_gsr = [self.data[key]['gsr'] for key in self.data]
        y_pos = np.arange(len(labels))
        print(labels)

        fig, ax = plt.subplots()
        fig.set_size_inches(40, 7.5)

        plt.bar(x + 0.00, values_bpm, color='r', width=0.25)
        plt.bar(x + 0.25, values_gsr, color='b', width=0.25)
        plt.xticks(x, labels)
        ax.set_title('GSR basing on valence and arousal')


        plt.show()

    def _group_by_emotion(self, data):
        emotions = {}
        for person in data:
            for video in person:
                if not video['bpm']:
                    continue
                e_name = emotion.get_class_for_values(video['valence'], video['arousal'])

                index = e_name
                if index not in emotions:
                    emotions[index] = {}
                    emotions[index]['bpm'] = []
                    emotions[index]['gsr'] = []

                # bpm_avg = mean([i[1] for i in video['bpm']])
                emotions[index]['bpm'].append(video['bpm'])
                emotions[index]['gsr'].append(video['gsr']/500)

        for key in emotions:
            emotions[key]['bpm'] = mean(emotions[key]['bpm'])
            emotions[key]['gsr'] = mean(emotions[key]['gsr'])

        return emotions
