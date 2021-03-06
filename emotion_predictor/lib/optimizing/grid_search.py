from sklearn.model_selection import GridSearchCV


class GridSearchOptimizer:
    def optimize(self, model, param_grid, x, y):
        """
        This method is a wrapper for sklearn built-in grid search method.
        Returns best parameter values from given param_grid for given dataset
        """
        gs = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=20, n_jobs=-1)
        gs.fit(x, y)
        return gs.best_score_, gs.best_params_
