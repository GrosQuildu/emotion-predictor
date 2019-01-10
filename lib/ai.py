import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from config import VALIDATION_SIZE, SEED, OPTIMIZED_ESTIMATORS
from lib.accuracy.reverse_sbs import ReverseSBS
from lib.accuracy.sbs import SBS
from lib.optimizing.grid_search import GridSearchOptimizer
from lib.prediction_analyser import PredictionAnalyser


class AI:
    """
    Contains most of the method connected to AI
    """
    def simple_test(self, x, y):
        """
        Basic accuracy test using RandomForestClassifier
        :param x: X list of samples
        :param y: Y list of samples
        """
        x_tr, x_val, y_tr, y_val = self._split_data(x, y)

        model = RandomForestClassifier()
        model.fit(x_tr, y_tr)
        predictions = model.predict(x_val)

        scored_value = accuracy_score(y_val, predictions)
        print(f"Accuracy score is: {scored_value}")

    def analyse_predictions(self, x, y):
        """
        Analyses what mistakes were made by model when predicting
        """
        x_tr, x_val, y_tr, y_val = self._split_data(x, y)
        model = SVC()
        model.fit(x_tr, y_tr)

        uniq_classes = set()
        for label in y_val:
            uniq_classes.add(label)

        print(f"Unique classes found: {uniq_classes}")

        analyser = PredictionAnalyser(model)
        return analyser.analyse(x_val, y_val)

    def sbs_score(self, x, y):
        """
        Selects best features using the SBS method
        """
        knn = KNeighborsClassifier(n_neighbors=2)
        sbs = SBS(knn, k_features=1)
        return sbs.fit(x, y)

    def reverse_sbs_score(self, x, y):
        """
        Selects best features using the sequence feature selection method
        """
        rev_sbs = ReverseSBS(SVC, {})
        return rev_sbs.calculate(x, y)

    def random_forest_score(self, x, y, labels):
        """
        Selects best features using the Random Forest method
        :param labels: labels of features
        """
        forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
        forest.fit(x, y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(len(x[0])):
            print("%2d %-*s %f" % (f + 1, 30, labels[indices[f]], importances[indices[f]]))

    def optimize_best_estimators(self, x, y):
        """
        Optimizes an estimator using the grid search method
        """
        svc_model, svc_params = OPTIMIZED_ESTIMATORS[0]
        dtc_model, dtc_params = OPTIMIZED_ESTIMATORS[3]

        optimizer = GridSearchOptimizer()

        param_grid = [
            {
                'n_estimators': [1, 2, 3, 4]
            }
        ]

        # in this particular case we are trying to optimize the AdaBoostClassifier using the best basic model
        return optimizer.optimize(
            AdaBoostClassifier(random_state=1, base_estimator=svc_model(**svc_params)),
            param_grid,
            x,
            y
        )

    def get_best_model(self, x, y):
        """
        Returns the best, trained model
        """
        model_class, params = OPTIMIZED_ESTIMATORS[1]
        model = model_class(**params)
        model.fit(x, y)

        return model

    def _split_data(self, x, y):
        """
        Spits data into train and validate sets
        """
        x_tr, x_val, y_tr, y_val = model_selection.train_test_split(x, y, test_size=VALIDATION_SIZE, random_state=SEED)

        return x_tr, x_val, y_tr, y_val
