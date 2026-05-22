from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def print_results(name, Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test):
    print(f"{name+":":<20}"
          f"{mean_squared_error(Y_train, Y_pred_train):>15.4f}"
          f"{mean_squared_error(Y_valid, Y_pred_valid):>15.4f}"
          f"{mean_squared_error(Y_test, Y_pred_test):>15.4f}"
          f"{r2_score(Y_train, Y_pred_train):>15.4f}"
          f"{r2_score(Y_valid, Y_pred_valid):>15.4f}"
          f"{r2_score(Y_test, Y_pred_test):>15.4f}")