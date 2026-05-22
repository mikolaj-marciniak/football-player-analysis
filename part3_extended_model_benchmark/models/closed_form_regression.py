from models.base_model import Model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.preprocessing import LabelEncoder
import numpy as np

class ClosedFormula(Model):
    def __init__(self, add_polynomial=False, classification=False):
        self.theta = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.add_polynomial = add_polynomial
        self.classification = classification
        self.encoder = None
        self.num_classes = None
        self.poly = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X_train, Y_train):
        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)

        if self.add_polynomial:
            self.poly = PolynomialFeatures(degree=5, include_bias=False)
            X_train = self.poly.fit_transform(X_train)

        X_train = np.c_[np.ones(X_train.shape[0]), X_train]

        if self.classification:
            self.encoder = LabelEncoder()
            Y_encoded = self.encoder.fit_transform(Y_train)
            self.num_classes = len(self.encoder.classes_)

            # One-vs-rest encoding for multi-class
            Y_ovr = np.eye(self.num_classes)[Y_encoded]

            try:
                self.theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_ovr
            except np.linalg.LinAlgError:
                print("Macierz osobliwa – nie można wyliczyć pseudoodwrotności.")
        else:
            Y_train = Y_train.values.reshape(-1, 1)
            try:
                self.theta = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train
            except np.linalg.LinAlgError:
                print("Macierz osobliwa – nie można wyliczyć pseudoodwrotności.")

    def predict(self, X_pred):
        X_pred = self.imputer.transform(X_pred)
        X_pred = self.scaler.transform(X_pred)

        if self.add_polynomial and self.poly is not None:
            X_pred = self.poly.transform(X_pred)

        X_pred = np.c_[np.ones(X_pred.shape[0]), X_pred]

        if self.classification:
            logits = X_pred @ self.theta
            probs = self.softmax(logits)
            preds_numeric = np.argmax(probs, axis=1)
            return self.encoder.inverse_transform(preds_numeric)
        else:
            return X_pred @ self.theta
