from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import numpy as np

def print_results(name, Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test):
    print(f"{name+":":<20}"
          f"{mean_squared_error(Y_train, Y_pred_train):>15.4f}"
          f"{mean_squared_error(Y_valid, Y_pred_valid):>15.4f}"
          f"{mean_squared_error(Y_test, Y_pred_test):>15.4f}"
          f"{r2_score(Y_train, Y_pred_train):>15.4f}"
          f"{r2_score(Y_valid, Y_pred_valid):>15.4f}"
          f"{r2_score(Y_test, Y_pred_test):>15.4f}")
    
def cross_val_results(name, mse_train, r2_train, mse_valid, r2_valid):
    print(f"{name:<20}{"MSE":>30}{"R^2":>30}")
    print(f"{"":<20}{"Train":>15}{"Valid":>15}{"Train":>15}{"Valid":>15}\n")
    for mse_t, r2_t, mse_v, r2_v in zip(mse_train, r2_train, mse_valid, r2_valid):
        print(f"{"":<20}{mse_t:>15.4f}{mse_v:>15.4f}{r2_t:>15.4f}{r2_v:>15.4f}")
    print(f"{"Średnia:":<20}"
          f"{np.mean(mse_train):>15.4f}"
          f"{np.mean(mse_valid):>15.4f}"
          f"{np.mean(r2_train):>15.4f}"
          f"{np.mean(r2_valid):>15.4f}")
