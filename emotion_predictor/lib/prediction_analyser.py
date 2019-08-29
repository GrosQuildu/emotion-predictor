import matplotlib.pyplot as plt
import numpy as np


class PredictionAnalyser:
    """
    Analyses predictions, showing which classes are predicted better and which are predicted worse.
    Calculates accuracy and shows plot
    """
    def __init__(self, model):
        self.model = model

    def analyse(self, x, y):
        all_sample_count = len(y)

        correct_sample_count = 0
        correct_samples = {}
        all_sample_count_dict = {}

        for i, sample in enumerate(x):
            prediction = self.model.predict([sample])[0]
            correct = y[i]

            self._increment_dict_value(all_sample_count_dict, correct)

            if prediction == correct:
                correct_sample_count += 1
                self._increment_dict_value(correct_samples, correct)

        main_accuracy = correct_sample_count / all_sample_count
        detail_accuracy = self._calculate_accuracy(correct_samples, all_sample_count_dict)
        self._make_plot(detail_accuracy)

        return main_accuracy, detail_accuracy

    def _increment_dict_value(self, container, key):
        if key in container:
            container[key] += 1
        else:
            container[key] = 1

    def _calculate_accuracy(self, correct_samples, all_samples):
        result = {}
        for key in all_samples:
            result[key] = correct_samples.pop(key, 0) / all_samples[key]

        return result

    def _make_plot(self, accuracy):
        values = [accuracy[key] for key in accuracy]
        labels = [key for key in accuracy]
        y_pos = np.arange(len(values))

        plt.bar(y_pos, values, align='center', alpha=0.5)
        plt.xticks(y_pos, labels)

        plt.show()
