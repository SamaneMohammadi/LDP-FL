"""
Model inversion attack for SER (Algorithm 3), adapted from Fredrikson et al.
(2015) from faces to speech features.

The adversary knows a target emotion label and has access to a client's model.
It starts from a blank feature vector and runs gradient descent on the *input*
to make the model as confident as possible in the target label, i.e. it
minimizes

    c(x) = 1 - f_label(x)

where f_label(x) is the model's predicted probability for the target emotion.
The reconstructed x is the attacker's guess at that client's speech features.

We then score the attack with the MSE between the reconstruction and the
client's real features. Under LDP the model is noisy, so the reconstruction is
poor and the MSE blows up - that large MSE is exactly the privacy protection the
paper reports in its attack table.
"""

import torch
import torch.nn.functional as F

import config


def invert(model, target_label, input_dim=config.INPUT_DIM,
           iters=config.ATTACK_ITERS, lr=config.ATTACK_LR,
           patience=config.ATTACK_PATIENCE, gamma=config.ATTACK_GAMMA,
           device="cpu"):
    """Reconstruct the input features that maximize the target label.

    Returns (best_x, best_cost).
    """
    model.eval()
    x = torch.zeros(1, input_dim, device=device, requires_grad=True)

    def cost(xv):
        # model outputs log-softmax; exp -> probability of the target label
        prob = model(xv).exp()[0, target_label]
        return 1.0 - prob

    best_x, best_cost, history = x.detach().clone(), float("inf"), []
    for _ in range(iters):
        c = cost(x)
        history.append(c.item())

        if c.item() < best_cost:
            best_cost, best_x = c.item(), x.detach().clone()

        # stop if no improvement over the last `patience` steps
        if len(history) > patience and c.item() >= max(history[-patience:]):
            break
        # stop if the cost is already past the threshold
        if c.item() <= 1 - gamma:
            break

        g, = torch.autograd.grad(c, x)
        with torch.no_grad():
            x = (x - lr * g).clamp(0, 1)   # Process(): keep features in [0,1]
        x.requires_grad_(True)

    return best_x, best_cost


def attack_mse(model, target_label, true_features, device="cpu"):
    """Run the attack and return MSE between reconstruction and real features."""
    recon, _ = invert(model, target_label, input_dim=true_features.numel(), device=device)
    return F.mse_loss(recon.flatten(), true_features.flatten().to(device)).item()
