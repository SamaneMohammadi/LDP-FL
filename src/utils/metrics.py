"""
Evaluation metrics: accuracy, macro-F1, precision, loss - the ones reported in
the paper's performance table.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    criterion = torch.nn.NLLLoss()
    y_true, y_pred, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(criterion(out, y).item())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_true.extend(y.cpu().numpy())

    if not y_true:
        return {"acc": 0.0, "f1": 0.0, "precision": 0.0, "loss": 0.0}

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "loss": float(np.mean(losses)),
    }
