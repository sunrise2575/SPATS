import torch
import numpy as np

class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: np.ndarray):
        self.mean = [X[..., i].mean() for i in range(X.shape[-1])]
        self.std = [X[..., i].std() for i in range(X.shape[-1])]

    def transform(self, X: np.ndarray) -> np.ndarray:
        result = []
        for i in range(X.shape[-1]):
            result.append((X[..., i] - self.mean[i]) / self.std[i])
        return np.stack(result, axis=-1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        result = []
        for i in range(X.shape[-1]):
            result.append(X[..., i] * self.std[i] + self.mean[i])
        return np.stack(result, axis=-1)

class StandardScaler_torch():
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X: torch.Tensor):
        self.mean = [X[..., i].mean() for i in range(X.shape[-1])]
        self.std = [X[..., i].std() for i in range(X.shape[-1])]

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        result = []
        for i in range(X.shape[-1]):
            result.append((X[..., i] - self.mean[i]) / self.std[i])
        return torch.stack(result, axis=-1)

    def inverse_transform(self, X: torch.Tensor) -> torch.Tensor:
        result = []
        for i in range(X.shape[-1]):
            result.append(X[..., i] * self.std[i] + self.mean[i])
        return torch.stack(result, axis=-1)