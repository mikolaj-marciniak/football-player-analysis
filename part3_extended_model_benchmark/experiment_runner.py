from models.closed_form_regression import ClosedFormula
from models.decision_tree_regression import DecisionTree
from models.gradient_descent_regression import GradientDescent
from models.linear_regression_model import LinearRegressionModel
from models.pytorch_linear_regression import PyTorchLinear
from models.svr_regression import SVM
from models.ensemble_regression import EnsembleModel
from models.mixture_of_experts import MixtureOfExperts
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import numpy as np

from sklearn.model_selection import train_test_split
from metrics_report import print_results
from metrics_report import cross_val_results
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report


import copy

def calculate(X, Y, cross_valid=False, add_polynomial=False, regularization=None, data_improvement=None, 
              grid_search=False, ensemble_method=None, mixture_of_experts=False):
    GD_alpha = 1e-3
    GD_epochs = 2000
    PT_lr = 0.01
    PT_epochs = 100

    # Utworzenie modeli
    models = {
         "LinReg": LinearRegressionModel(regularization=None),
         "DecisionTree": DecisionTree(grid_search),
         "SVR": SVM(grid_search),
         "ClosedFormula": ClosedFormula(add_polynomial=False, classification=False),
         "GradDes": GradientDescent(GD_alpha, GD_epochs, add_polynomial=False, regularization=None),
         "CPU(LinReg)": PyTorchLinear('cpu', PT_lr, PT_epochs),
         "GPU(LinReg)": PyTorchLinear('cuda', PT_lr, PT_epochs),
         "MixtureOfExperts": MixtureOfExperts()
    }

    if data_improvement is not None:
        Y = Y.apply(categorize_player)
        if data_improvement == "Oversampling":
            smote = SMOTE(k_neighbors=1, random_state=42)
            X, Y = smote.fit_resample(X, Y)
        if data_improvement == "Undersampling":
            undersampler = NearMiss(k_neighbors=1, version=1)
            X, Y = undersampler.fit_resample(X, Y)

    if ensemble_method is not None:
        models["Ensemble " + ensemble_method] = EnsembleModel(ensemble_method)
    if mixture_of_experts:
        models["MixtureOfExperts"] = MixtureOfExperts()

    if cross_valid:
        crossValid(X, Y, models)        
    else:
        normalValid(X, Y, models)

def crossValid(X, Y, models):
    # Podział na zbiory
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Cross Validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    dict_m = dict()
    for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            Y_tr, Y_val = Y_train.iloc[train_idx], Y_train.iloc[val_idx]

            new_models = {key: copy.deepcopy(model) for key, model in models.items()}
            result = true_calc(new_models, X_tr, Y_tr, None, None, X_val, Y_val)

            for name, tuple in result.items():
                y_pred_train = tuple[0]
                y_pred_valid = tuple[2]
                if name not in dict_m:
                    dict_m[name] = {"mse_train": [], "r2_train": [], "mse_valid": [], "r2_valid": []}
                dict_m[name]["mse_train"].append(mean_squared_error(Y_tr, y_pred_train))
                dict_m[name]["r2_train"].append(r2_score(Y_tr, y_pred_train))
                dict_m[name]["mse_valid"].append(mean_squared_error(Y_val, y_pred_valid))
                dict_m[name]["r2_valid"].append(r2_score(Y_val, y_pred_valid))
    for name, pom in dict_m.items():
        cross_val_results(name, pom["mse_train"], pom["r2_train"], pom["mse_valid"], pom["r2_valid"])
                

def normalValid(X, Y, models):
    # Wypisanie kolumn tabelki
    print()
    print(f"{"":<20}{"MSE":>15}{"R^2":>45}")
    print(f"{"":<20}{"Train":>15}{"Valid":>15}{"Test":>15}{"Train":>15}{"Valid":>15}{"Test":>15}")
    print()

    # Podział na zbiory
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    result = true_calc(models, X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    for name, tuple in result.items():
        i=0
        print_results(name, Y_train, tuple[0], Y_valid, tuple[1], Y_test, tuple[2])
        #print(classification_report(Y_test, tuple[2]))

def true_calc(models, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    result = dict()
     # Obliczenie dla poszczególnych modeli
    for name, model in models.items():
        if isinstance(model, (GradientDescent, PyTorchLinear)):
            model.fit(X_train, Y_train, X_valid, Y_valid)
        else:
            model.fit(X_train, Y_train)

        Y_pred_train = model.predict(X_train)

        if X_valid is None:
            Y_pred_valid = None
        else:
            Y_pred_valid = model.predict(X_valid)

        Y_pred_test = model.predict(X_test)

        result[name] = (Y_pred_train, Y_pred_valid, Y_pred_test)
    return result

def categorize_player(overall):
    if overall <= 65:
        return "slaby"
    elif overall <= 80:
        return "sredni"
    else:
        return "dobry"
