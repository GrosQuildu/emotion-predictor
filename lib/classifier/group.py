from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from config import OPTIMIZED_ESTIMATORS
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder


# class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
#     def __init__(self, classifiers, vote='classlabel', weights=None):
#         self.classifiers = classifiers
#         self.named_classifiers = {
#             key: value for key, value in _name_estimators(classifiers)
#         }
#         self.vote = vote
#         self.weights = weights
#         self.label_enc = None
#         self.classes = None
#         self._classifiers = None
#
#     def fit(self, x, y):
#         self.label_enc = LabelEncoder()
#         self.label_enc.fit(y)
#         self.classes = self.label_enc.classes_
#         self._classifiers = []
#
#         for clf in self.classifiers:
#             fitted_clf = clone(clf).fit(x, self.label_enc.transform(y))
#             self._classifiers.append(fitted_clf)
#
#         return self
#
#     def predict(self, x):
#         if self.vote == 'probability':
#             maj_vote = np.argmax(self.predict_proba(x), axis=1)
#         else:
#             predictions = np.asarray([clf.predict(x) for clf in self._classifiers])
#             maj_vote = np.apply_along_axis(
#                 lambda value: np.argmax(np.bincount(value, weights=self.weights)),
#                 axis=1,
#                 arr=predictions
#             )
#
#         maj_vote = self.label_enc.inverse_transform(maj_vote)
#         return maj_vote
#
#     def predict_proba(self, x):
#         probas = np.asarray([clf.predict_proba(x) for clf in self._classifiers])
#         avg_proba = np.average(probas, axis=0, weights=self.weights)
#         return avg_proba
#
#     def get_params(self, deep=True):
#         if deep:
#             out = self.named_classifiers.copy()
#             for name, step in iter(self.named_classifiers):
#                 for key, value in iter(step.get_params(deep=True)):
#                     out[f"{name}__{key}"] = value
#             return out
#         else:
#             return super(MajorityVoteClassifier, self).get_params(deep=False)

def create_majority_voting_classifier():
    estimators = []
    for model, params in OPTIMIZED_ESTIMATORS:
        estimators.append((model.__name__, model(**params)))

    return VotingClassifier(
        estimators=estimators,
        flatten_transform=True,
        n_jobs=-1
    )


def create_ada_boosted_classifier():
    model, params = OPTIMIZED_ESTIMATORS[0]
    estimator = model(**params)

    return AdaBoostClassifier(
        random_state=1,
        base_estimator=estimator
    )
