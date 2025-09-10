import torch
import torch.nn as nn
import torch.nn.functional as F
from cliffordlayers.nn.modules.cliffordlinear import CliffordLinear


class FFLayer(nn.Module):
    def __init__(self, g,  in_channels : int, out_channels : int,
                   lr :int, threshold : int = 1.0,   bias : bool = True):
        super().__init__()
        self.g = g
        self.bias = bias
        self.lr = lr 
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.threshold = threshold
        self.layer = CliffordLinear(g = self.g, in_channels= self.in_channels,
                                      out_channels= self.out_channels)
        self.optmizer = torch.optim.Adam(self.layer.parameters(), lr = self.lr)
    
    def forward(self, x):
        return F.relu(self.layer(x))

    def goodness(self, activations):
        return (activations**2).mean(dim=1)

    def train_step(self, x_pos, x_neg, optimizer):
        act_pos = self.forward(x_pos)
        act_neg = self.forward(x_neg)

        g_pos = self.goodness(act_pos)
        g_neg = self.goodness(act_neg)

        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()

        loss = loss_pos + loss_neg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return act_pos, act_neg, loss.item()
    
class FFOutLayer(nn.Module):
    def __init__(self, g,  in_channels : int, out_channels : int,
                 lr :int,threshold : int = 2.0 ,    bias : bool = True):
        super().__init__()
        self.g = g
        self.bias = bias
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.threshold = threshold
        self.layer = CliffordLinear(g = self.g, in_channels= self.in_channels,
                                      out_channels= self.out_channels)
        self.optmizer = torch.optim.Adam(self.layer.parameters(), lr = self.lr)
    
    def forward(self, x):
        return self.layer(x)

    def goodness(self, score):
        """Goodness for last layer = directly the scalar output"""
        return score.squeeze(1)  # (batch,)

    def train_step(self, x_pos, x_neg):
        act_pos = self.forward(x_pos)
        act_neg = self.forward(x_neg)

        g_pos = self.goodness(act_pos)
        g_neg = self.goodness(act_neg)

        loss_pos = F.softplus(self.threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - self.threshold).mean()

        loss = loss_pos + loss_neg

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return  loss.item()