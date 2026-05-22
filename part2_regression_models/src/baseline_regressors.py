from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from metrics_report import print_results

def z1(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    models = {
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'SVM': SVR()
    }

    for name, model in models.items():
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, Y_train)
        Y_pred_train = pipeline.predict(X_train)
        Y_pred_valid = pipeline.predict(X_valid)
        Y_pred_test = pipeline.predict(X_test)
        print_results(name, Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test)
