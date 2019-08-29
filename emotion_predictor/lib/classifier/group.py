from sklearn.ensemble import VotingClassifier, AdaBoostClassifier
from emotion_predictor.config import OPTIMIZED_ESTIMATORS


def create_majority_voting_classifier():
    """
    Returns a majority voting classifier by connecting all optimized simple classifiers
    :return:
    """
    estimators = []
    for model, params in OPTIMIZED_ESTIMATORS:
        estimators.append((model.__name__, model(**params)))

    return VotingClassifier(
        estimators=estimators,
        flatten_transform=True,
        n_jobs=-1
    )


def create_ada_boosted_classifier():
    """
    Returns AdaBoostClassifier created by connecting optimized SVC models
    :return:
    """
    model, params = OPTIMIZED_ESTIMATORS[0]
    estimator = model(**params)

    return AdaBoostClassifier(
        random_state=1,
        base_estimator=estimator
    )
