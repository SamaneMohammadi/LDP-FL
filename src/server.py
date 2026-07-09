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
