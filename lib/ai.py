import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lib.accuracy.sbs import SBS
from lib.accuracy.reverse_sbs import ReverseSBS


VALIDATION_SIZE = 0.2
SEED = 5


class AI:
    def load_data(self, x, y):
        x_tr, x_val, y_tr, y_val = self._split_data(x, y)
        self._x_tr = x_tr
        self._x_val = x_val
        self._y_tr = y_tr
        self._y_val = y_val

    def test(self):
        #model = RandomForestClassifier(max_depth=8, n_estimators=100, max_features='auto', min_samples_split=3)
        #model = RandomForestClassifier()
        model = SVC()
        #model = MLPClassifier()

        model.fit(self._x_tr, self._y_tr)
        predictions = model.predict(self._x_val)

        scored_value = accuracy_score(self._y_val, predictions)
        print(f"Accuracy score is: {scored_value}")

    def sbs_score(self, x, y):
        knn = KNeighborsClassifier(n_neighbors=2)
        sbs = SBS(knn, k_features=1)
        return sbs.fit(x, y)

    def reverse_sbs_score(self, x, y):
        rev_sbs = ReverseSBS(SVC)
        return rev_sbs.fit(x, y)

    def random_forest_score(self, x, y, labels):
        forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        forest.fit(x, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(len(x[0])):
            print("%2d %-*s %f" % (f + 1, 30, labels[indices[f]], importances[indices[f]]))

    def _split_data(self, x, y):
        x_tr, x_val, y_tr, y_val = model_selection.train_test_split(x, y, test_size=VALIDATION_SIZE, random_state=SEED)

        return x_tr, x_val, y_tr, y_val