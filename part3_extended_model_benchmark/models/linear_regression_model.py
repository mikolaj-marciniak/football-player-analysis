from models.base_model import Model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class LinearRegressionModel(Model):
    def __init__(self, regularization):
        self.regularization = regularization
        model = None
        if regularization == "Ridge":
            model = Ridge(alpha=1)
        elif regularization == "Lasso":
            model = Lasso(alpha=1)
        else:
            model = LinearRegression()

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])

    def fit(self, X_train, Y_train):
        self.pipeline.fit(X_train, Y_train)

        #print(f"\nWagi(LinReg)({self.regularization or 'None'}):", self.pipeline.named_steps['model'].coef_)

    def predict(self, X_pred):
        Y_pred = self.pipeline.predict(X_pred)
        return Y_pred
