import pandas as pd
from sklearn.model_selection import train_test_split

from experiment_runner import calculate

def main():
    # Wczytanie i porządkowanie danych
    df = pd.read_csv('players_22.csv', low_memory=False)
    df = df[df['player_positions'] != 'GK']
    features = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
    X = df[features]
    Y = df['overall']

    #Obliczanie ze wskazanymi parametrami
    calculate(X, Y, cross_valid=False, add_polynomial=False, regularization=None, grid_search=False,
              ensemble_method=None, mixture_of_experts=False, data_improvement=None)

if __name__ == "__main__":
    main()
