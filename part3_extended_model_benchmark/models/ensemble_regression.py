from models.base_model import Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor

class EnsembleModel(Model):
    def __init__(self, ensemble_method=None):
        self.ensemble_method = ensemble_method
        if self.ensemble_method is None:
            return
        
        preprocessing = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

        lin_reg = Pipeline([('pre', preprocessing), ('model', LinearRegression())])
        tree = Pipeline([('pre', preprocessing), ('model', DecisionTreeRegressor())])
        svr = Pipeline([('pre', preprocessing), ('model', SVR())])


        if self.ensemble_method == "Voting":
            self.pipeline = VotingRegressor(estimators=[('lr', lin_reg),('dt', tree),('svr', svr)])
        if self.ensemble_method == "Stacking":
            self.pipeline = StackingRegressor(estimators=[('lr', lin_reg),('dt', tree),('svr', svr)])

    def fit(self, X_train, Y_train):
        self.pipeline.fit(X_train, Y_train)

    def predict(self, X_pred):
        Y_pred = self.pipeline.predict(X_pred)
        return Y_pred
