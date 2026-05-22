from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import torch.optim as optim

import time

from metrics_report import print_results

def z3(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    imputer = SimpleImputer()
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)

    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test.values.reshape(-1, 1), dtype=torch.float32)
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    Y_valid = torch.tensor(Y_valid.values.reshape(-1, 1), dtype=torch.float32)

    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

    class LinearRegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
        def forward(self, x):
            return self.linear(x)
        
    devices = {
        'CPU': 'cpu',
        'GPU': 'cuda'
    }
    input_dim = X_train.shape[1]
    for name, device in devices.items():
        if device=='cuda' and not torch.cuda.is_available():
            print("GPU niedostępne")
        else:
            model = LinearRegressionModel(input_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            start_time = time.time()
            epochs = 100
            best_val_loss = float('inf')
            for epoch in range(epochs):
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_valid.to(device))
                    val_loss = criterion(val_outputs, Y_valid.to(device))
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_model_state = model.state_dict()
                model.train()
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            end_time = time.time()
            model.eval()
            with torch.no_grad():
                Y_pred_train = model(X_train.to(device)).cpu().numpy()
                Y_pred_valid = model(X_valid.to(device)).cpu().numpy()
                Y_pred_test = model(X_test.to(device)).cpu().numpy()
            print_results(f"{name + "-LinReg"}({end_time - start_time:.2f}s)", Y_train, Y_pred_train, Y_valid, Y_pred_valid, Y_test, Y_pred_test)
