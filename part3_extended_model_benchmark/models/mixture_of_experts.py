from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class MixtureOfExperts:
    def __init__(self):
        self.experts = [
            LinearRegression(),
            DecisionTreeRegressor(),
            SVR()
        ]
        self.gating_model = LinearRegression()
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def fit(self, X, Y):
        if isinstance(Y, pd.Series):
            Y = Y.values

        X = self.imputer.fit_transform(X)
        X = self.scaler.fit_transform(X)

        preds = []
        self.trained_experts = []
        for model in self.experts:
            m = model
            m.fit(X, Y)
            self.trained_experts.append(m)
            preds.append(m.predict(X))

        preds = np.array(preds).T

        self.gating_model.fit(X, preds)

    def predict(self, X):
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        gating_raw = self.gating_model.predict(X)
        gating_weights = self.softmax(gating_raw)

        expert_preds = np.column_stack([m.predict(X) for m in self.trained_experts])
        weighted_preds = np.sum(gating_weights * expert_preds, axis=1)
        return weighted_preds
