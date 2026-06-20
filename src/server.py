"""
Server side of LDP-FL (Algorithm 1, lines 13-14).

The server averages the noisy client gradients (FedSGD) and steps the global
model:

    g_tilde = (1/K) sum_i g_tilde_i
    w_{t+1} = w_t - eta * g_tilde

It only ever sees noisy gradients, never raw data or clean updates.
"""

import torch


class Server:
    def aggregate_and_update(self, global_weights, client_grads, lr):
        """Average the client gradients and take one global FedSGD step."""
        k = len(client_grads)
        # mean gradient per parameter across the K clients
        mean_grad = [
            torch.stack([client_grads[i][p] for i in range(k)]).mean(dim=0)
            for p in range(len(global_weights))
        ]
        return [w - lr * g for w, g in zip(global_weights, mean_grad)]
