import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from config import CPU_CORES_NUM


class MultimodelCrossValidator:
    def __init__(self, x_train, y_train, estimators):
        self.x_train = x_train
        self.y_train = y_train
        self.estimators = estimators

    def validate_all(self):
        scores = []
        for estimator in self.estimators:
            estimator_name = estimator.__name__
            print(f"Validating {estimator_name}")
            mean, std = self._validate(estimator, n_threads=CPU_CORES_NUM, n_subsets=20)
            scores.append((mean, estimator_name))

        return scores

    def _validate(self, estimator, n_threads=1, n_subsets=10):
        pipe_lr = Pipeline([
            ('scl', StandardScaler()),
            ('pca', PCA(n_components=2)),
            ('clf', estimator())
        ])

        scores = cross_val_score(estimator=pipe_lr, X=self.x_train, y=self.y_train, cv=n_subsets, n_jobs=n_threads)
        mean = np.mean(scores)
        std = np.std(scores)
        print("Results of cross validation: ", scores)
        print(f"Accuracy: mean {mean}, std {std}")
        return mean, std
