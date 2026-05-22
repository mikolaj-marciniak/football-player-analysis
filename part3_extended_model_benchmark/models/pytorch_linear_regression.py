from models.base_model import Model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import time

class PyTorchLinear(Model):
    def __init__(self, device, lr, epochs):
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.time = None
        self.imputer = SimpleImputer()
        self.scaler = StandardScaler()

    def fit(self, X_train, Y_train, X_valid, Y_valid):
        best_model_state = None
        X_train = self.imputer.fit_transform(X_train)
        X_train = self.scaler.fit_transform(X_train)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        Y_train = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32)

        if X_valid is not None:
            X_valid = self.imputer.transform(X_valid)
            X_valid = self.scaler.transform(X_valid)
            X_valid = torch.tensor(X_valid, dtype=torch.float32)
            Y_valid = torch.tensor(Y_valid.values.reshape(-1, 1), dtype=torch.float32)

        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)
        input_dim = X_train.shape[1]

        if self.device=='cuda' and not torch.cuda.is_available():
            print("GPU niedostępne")
        else:
            self.model = LinearRegressionModel(input_dim).to(self.device)
            criterion = nn.MSELoss()
            optimizer = optim.SGD(self.model.parameters(), lr=0.01)
            start_time = time.time()
            best_val_loss = float('inf')
            for epoch in range(self.epochs):
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                if X_valid is not None:
                    self.model.eval()
                    with torch.no_grad():
                        val_outputs = self.model(X_valid.to(self.device))
                        val_loss = criterion(val_outputs, Y_valid.to(self.device))
                        if val_loss.item() < best_val_loss:
                            best_val_loss = val_loss.item()
                            best_model_state = self.model.state_dict()
                    self.model.train()
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
            end_time = time.time()
            self.time = end_time - start_time

    def predict(self, X_pred):
        X_pred = self.imputer.transform(X_pred)
        X_pred = self.scaler.transform(X_pred)
        X_pred = torch.tensor(X_pred, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            Y_pred = self.model(X_pred.to(self.device)).cpu().numpy()
        return Y_pred


class LinearRegressionModel(nn.Module):
            def __init__(self, input_dim):
                super(LinearRegressionModel, self).__init__()
                self.linear = nn.Linear(input_dim, 1)
            def forward(self, x):
                return self.linear(x)
