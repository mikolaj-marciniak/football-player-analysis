from models.base_model import Model
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

class SVM(Model):
    def __init__(self, grid_search=False):
        self.grid_search = grid_search
        if grid_search:
            param_grid = {
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.2, 0.5],
                'kernel': ['linear', 'rbf']
            }
            model = GridSearchCV(SVR(), param_grid, scoring='r2', cv=3)
        else:
            model = SVR()

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    def fit(self, X_train, Y_train):
        self.pipeline.fit(X_train, Y_train)
        if self.grid_search:
            model = self.pipeline.named_steps['model']
            print("Najlepsze parametry:", model.best_params_)
            print("\nWyniki dla parametrów (SVR):")
            for r2, params in zip(model.cv_results_['mean_test_score'], model.cv_results_['params']):
                print(f"R^2: {round(r2, 4):.4f} | Parametry: {params}")

    def predict(self, X_pred):
        Y_pred = self.pipeline.predict(X_pred)
        return Y_pred
