import numpy as np
from sklearn.model_selection import cross_val_score
from emotion_predictor.config import CPU_CORES_NUM
from emotion_predictor.lib.classifier.group import create_majority_voting_classifier


class MultimodelCrossValidator:
    def __init__(self, x_train, y_train, estimators):
        self.x_train = x_train
        self.y_train = y_train
        self.estimators = estimators

    def validate_all(self):
        """
        Performs cross-validation on all given estimators and returns their scores
        """
        scores = []
        for estimator in self.estimators:
            if isinstance(estimator, tuple):
                estimator_class = estimator[0]
                params = estimator[1]
            else:
                estimator_class = estimator
                params = {}
            estimator_name = estimator_class.__name__
            print(f"Validating {estimator_name}")
            mean, std = self._validate(estimator_class(**params), n_threads=CPU_CORES_NUM, n_subsets=20)
            scores.append((mean, estimator_name))

        mean, std = self._validate(create_majority_voting_classifier(), n_threads=CPU_CORES_NUM, n_subsets=20)
        scores.append((mean, 'VotingClassifier'))

        return scores

    def _validate(self, estimator, n_threads=1, n_subsets=20):
        scores = cross_val_score(estimator=estimator, X=self.x_train, y=self.y_train, cv=n_subsets, n_jobs=n_threads)
        mean = np.mean(scores)
        std = np.std(scores)
        print("Results of cross validation: ", scores)
        print(f"Accuracy: mean {mean}, std {std}")
        return mean, std
