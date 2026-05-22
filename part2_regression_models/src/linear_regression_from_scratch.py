from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import numpy as np

from metrics_report import print_results


def z2(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_valid = imputer.transform(X_valid)
    X_valid = scaler.transform(X_valid)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    Y_train = Y_train.values.reshape(-1, 1)
    Y_valid = Y_valid.values.reshape(-1, 1)
    Y_test = Y_test.values.reshape(-1, 1)

    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_valid = np.c_[np.ones(X_valid.shape[0]), X_valid]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    # Zamknięta formuła
    try:
        theta_closed = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ Y_train
        Y_pred_train = X_train @ theta_closed
        Y_pred_valid = X_valid @ theta_closed
        Y_pred_test = X_test @ theta_closed
        print_results("ClosedForm", Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test)
    except np.linalg.LinAlgError:
        print("Zamknięta formuła: macierz X^TX jest osobliwa i nie można wyliczyć pseudoodwrotności.")

    # Gradient Descent
    alpha = 1e-3
    epochs = 10000
    m = X_train.shape[0]
    n = X_train.shape[1]
    theta_gd = np.zeros((n, 1))
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        gradients = (2/m) * X_train.T @ (X_train @ theta_gd - Y_train)
        theta_gd -= alpha * gradients

        Y_pred_valid = X_valid @ theta_gd
        valid_loss = mean_squared_error(Y_valid, Y_pred_valid)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_theta = theta_gd.copy()

    Y_pred_train = X_train @ best_theta
    Y_pred_valid = X_valid @ best_theta
    Y_pred_test = X_test @ best_theta
    print_results("Gradient Descent", Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test)
