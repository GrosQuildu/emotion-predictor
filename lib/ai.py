import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from lib.accuracy.sbs import SBS
from lib.accuracy.reverse_sbs import ReverseSBS
from lib.prediction_analyser import PredictionAnalyser


VALIDATION_SIZE = 0.2
SEED = 1


class AI:
    def load_data(self, x, y):
        x_tr, x_val, y_tr, y_val = self._split_data(x, y)
        self._x_tr = x_tr
        self._x_val = x_val
        self._y_tr = y_tr
        self._y_val = y_val

    def test(self):
        #model = RandomForestClassifier(max_depth=8, n_estimators=100, max_features='auto', min_samples_split=3)
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.05)
        # model = SVC()
        #model = MLPClassifier()

        model.fit(self._x_tr, self._y_tr)
        predictions = model.predict(self._x_val)

        scored_value = accuracy_score(self._y_val, predictions)
        print(f"Accuracy score is: {scored_value}")

    def split_predictions(self):
        model = SVC()
        model.fit(self._x_tr, self._y_tr)

        uniq_classes = set()
        for label in self._y_val:
            uniq_classes.add(label)

        print(f"Unique classes found: {uniq_classes}")

        analyser = PredictionAnalyser(model)
        return analyser.analyse(self._x_val, self._y_val)

    def sbs_score(self, x, y):
        knn = KNeighborsClassifier(n_neighbors=2)
        sbs = SBS(knn, k_features=1)
        return sbs.fit(x, y)

    def reverse_sbs_score(self, x, y):
        # rev_sbs = ReverseSBS(LogisticRegression)
        rev_sbs = ReverseSBS(MLPClassifier, {})
        return rev_sbs.calculate(x, y)

    def random_forest_score(self, x, y, labels):
        forest = RandomForestClassifier(criterion='entropy')
        forest.fit(x, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(len(x[0])):
            print("%2d %-*s %f" % (f + 1, 30, labels[indices[f]], importances[indices[f]]))

    def _split_data(self, x, y):
        x_tr, x_val, y_tr, y_val = model_selection.train_test_split(x, y, test_size=VALIDATION_SIZE, random_state=SEED)

        return x_tr, x_val, y_tr, y_val