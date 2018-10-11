import operator
import sys
from sklearn import model_selection
from sklearn.metrics import accuracy_score


class ReverseSBS:
    def __init__(self, estimator, scoring=accuracy_score, test_size=0.20, random_state=1):
        self.estimator = estimator
        self.scoring = scoring
        self.test_size = test_size
        self.random_state = random_state

    def calculate(self, x, y):
        number_of_features = len(x[0])

        results = []
        for i in range(number_of_features):
            keys, accuracy = self._fit(x, y, starting_feature=i)
            results.append((keys, accuracy))

        max_accuracy = -1
        max_accuracy_keys = None
        for i in results:
            if i[1] > max_accuracy:
                max_accuracy = i[1]
                max_accuracy_keys = i[0]

        print(max_accuracy_keys)
        print(max_accuracy)
        sys.exit(0)
        return max_accuracy_keys, max_accuracy

    def _fit(self, x, y, starting_feature=0):
        max_features = len(x[0]) #number of all features
        remaining_features = [i for i in range(0, max_features)] #indices of remaining features (without 0=bpm)
        remaining_features.remove(starting_feature) #in order not to include the same feature twice
        x_best = [[i] for i in x[:, starting_feature]] #initial best feature configuration
        result_features_keys = [starting_feature] #keys of best features
        accuracy_best = self._test_accuracy(x_best, y)

        while remaining_features:
            accuracy_results = {}
            for i in remaining_features:
                x_test = self._add_lists(x_best, x[:, i])
                accuracy = self._test_accuracy(x_test, y)
                accuracy_results[i] = accuracy

            max_index, max_value = self._get_max_features(accuracy_results)

            if max_value < accuracy_best:
                break

            x_best = self._add_lists(x_best, x[:, max_index])
            remaining_features.remove(max_index)
            result_features_keys.append(max_index)
            accuracy_best = max_value

        return result_features_keys, accuracy_best

    def _test_accuracy(self, x, y):
        x_train, x_validate, y_train, y_validate = self._split_data(x, y)
        model = self.estimator()
        model.fit(x_train, y_train)
        predictions = model.predict(x_validate)

        return accuracy_score(y_validate, predictions)

    def _split_data(self, x, y):
        x_tr, x_val, y_tr, y_val = model_selection.train_test_split(
            x,
            y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        return x_tr, x_val, y_tr, y_val

    def _add_lists(self, x1, x2):
        if len(x1) != len(x2):
            raise Exception("Lists must be the same length")

        result = x1.copy()
        for i in range(len(result)):
            result[i].append(x2[i])

        return result

    def _get_max_features(self, accuracy_results):
        max_index = -1
        max_value = -1

        for index, value in accuracy_results.items():
            if value > max_value:
                max_value = value
                max_index = index

        return max_index, max_value
