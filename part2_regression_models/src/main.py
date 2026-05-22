import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Wczytanie i porządkowanie danych
    df = pd.read_csv('players_22.csv', low_memory=False)
    df = df[df['player_positions'] != 'GK']
    features = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    X = df[features]
    Y = df['overall']

    # Podział na zbiory
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Wypisanie kolumn tabelki
    print()
    print(f"{"":<20}{"MSE":>15}{"R^2":>45}")
    print(f"{"":<20}{"Train":>15}{"Valid":>15}{"Test":>15}{"Train":>15}{"Valid":>15}{"Test":>15}")
    print()

    # Zad1
    from baseline_regressors import z1
    z1(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    # Zad2
    from linear_regression_from_scratch import z2
    z2(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

    # Zad3
    from pytorch_linear_regression import z3
    z3(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)

if __name__ == "__main__":
    main()
