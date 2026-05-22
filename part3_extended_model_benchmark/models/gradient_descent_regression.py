from models.base_model import Model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

class GradientDescent(Model):
    def __init__(self, alpha, epochs, add_polynomial=False, regularization=None):
        self.alpha = alpha
        self.epochs = epochs
        self.add_polynomial = add_polynomial
        self.poly = None
        self.theta = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()
        self.loss_train = []
        self.loss_valid = []
        self.regularization = regularization

    def fit(self, X_train, Y_train, X_valid, Y_valid):
        if self.add_polynomial:
            self.poly = PolynomialFeatures(degree=5, include_bias=False)
            X_train = self.poly.fit_transform(X_train)
            if X_valid is not None:
                X_valid = self.poly.transform(X_valid)

        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)
        X_train = np.c_[np.ones(X_train.shape[0]), X_train]
        Y_train = Y_train.values.reshape(-1, 1)

        if X_valid is not None:
            X_valid = self.imputer.transform(X_valid)
            X_valid = self.scaler.transform(X_valid)
            X_valid = np.c_[np.ones(X_valid.shape[0]), X_valid]
            Y_valid = Y_valid.values.reshape(-1, 1)

        m = X_train.shape[0]
        n = X_train.shape[1]
        theta_gd = np.zeros((n, 1))
        best_valid_loss = float('inf')

        reg_lambda = 1

        for epoch in range(self.epochs):
            gradients = (2/m) * X_train.T @ (X_train @ theta_gd - Y_train)

            if self.regularization == 'Ridge':
                reg_term = 2 * reg_lambda * np.r_[[[0]], theta_gd[1:]]
                gradients += reg_term
            elif self.regularization == 'Lasso':
                reg_term = reg_lambda * np.r_[[[0]], np.sign(theta_gd[1:])]
                gradients += reg_term

            theta_gd -= self.alpha * gradients

            Y_pred_train = X_train @ theta_gd
            train_loss = mean_squared_error(Y_train, Y_pred_train)
            self.loss_train.append(train_loss)

            if X_valid is not None:
                Y_pred_valid = X_valid @ theta_gd
                valid_loss = mean_squared_error(Y_valid, Y_pred_valid)
                self.loss_valid.append(valid_loss)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_theta = theta_gd.copy()

        if X_valid is not None:
            self.theta = best_theta
        else:
            self.theta = theta_gd

        self.wykres()
        #print(f"\nWagi(GradientD)({self.regularization or 'None'}):", self.theta.flatten())

    def predict(self, X_pred):
        if self.add_polynomial and self.poly is not None:
            X_pred = self.poly.transform(X_pred)

        X_pred = self.imputer.transform(X_pred)
        X_pred = self.scaler.transform(X_pred)
        X_pred = np.c_[np.ones(X_pred.shape[0]), X_pred]
        
        Y_pred = X_pred @ self.theta
        return Y_pred
    
    def wykres(self):
        plt.plot(self.loss_train, label="Train MSE")
        plt.plot(self.loss_valid, label="Validation MSE")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.title("Loss per Epoch")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight')
