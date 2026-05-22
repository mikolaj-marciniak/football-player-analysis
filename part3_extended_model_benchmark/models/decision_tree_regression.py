from models.base_model import Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class DecisionTree(Model):
    def __init__(self, grid_search=False):
        self.grid_search = grid_search
        if grid_search:
            tree_params = {
                'max_depth': [2, 4, 6, 8, 10, None],
                'min_samples_split': [2, 5, 10]
            }
            model = GridSearchCV(DecisionTreeRegressor(), tree_params, scoring='r2', cv=3)
        else:
            model = DecisionTreeRegressor()

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    def fit(self, X_train, Y_train):
        self.pipeline.fit(X_train, Y_train)
        if self.grid_search:
            model = self.pipeline.named_steps['model']
            best_params = model.best_params_
            print("Najlepsze parametry:", best_params)
            print("\nWyniki dla parametrów (DecisionTree):")
            for mean_score, params in zip(model.cv_results_['mean_test_score'], model.cv_results_['params']):
                print(f"R^2: {round(mean_score, 4):.4f} | Parametry: {params}")

    def predict(self, X_pred):
        Y_pred = self.pipeline.predict(X_pred)
        return Y_pred
