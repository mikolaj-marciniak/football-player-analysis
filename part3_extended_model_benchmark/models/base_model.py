from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def fit(self):
        pass
    def predict(self):
        pass