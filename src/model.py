"""
SER model: a small MLP, exactly as in the paper - two dense layers [256, 128]
with ReLU and 0.2 dropout, then a 4-way classifier over the 988-dim OpenSMILE
emobase features. Outputs log-softmax (paired with NLL loss).
"""

import torch
import torch.nn as nn

import config


class MLP(nn.Module):
    def __init__(self, input_dim=config.INPUT_DIM, hidden=config.HIDDEN_DIMS,
                 num_classes=config.NUM_CLASSES, dropout=config.DROPOUT):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden[0])
        self.dense2 = nn.Linear(hidden[0], hidden[1])
        self.pred = nn.Linear(hidden[1], num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = x.float()
        if x.dim() > 2:
            x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.relu(self.dense1(x)))
        x = self.dropout(self.relu(self.dense2(x)))
        return torch.log_softmax(self.pred(x), dim=1)


def build_model():
    return MLP()
