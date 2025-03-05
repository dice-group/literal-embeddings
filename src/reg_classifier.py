import torch
import torch.nn as nn


class RegressionClassifier(nn.Module):
    def __init__(self, ent_dim, num_attr, threshold=0.77):
        super(RegressionClassifier, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(ent_dim, num_attr))
        nn.init.xavier_uniform_(self.weights)
        self.threshold = threshold

    def forward(self, x):
        return torch.sigmoid(torch.matmul(x, self.weights))

    def predict(self, x, r=None):
        with torch.no_grad():
            probabilities = self.forward(x)
            predictions = (probabilities >= self.threshold).float()
            if r:
                return predictions[r]
            return predictions
