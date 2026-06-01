import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []
        with torch.no_grad():
            for i, layer in enumerate(model):
                x = layer(x)
                if isinstance(layer, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)
                    dead_fraction = round((x <= 0).all(dim=0).float().mean().item(), 4)

                    stats.append({
                        'mean': mean,
                        'std': std,
                        'dead_fraction': dead_fraction
                    })

        return stats


    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        out = model(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        stats = []
        for i, layer in enumerate(model):
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)

                stats.append({
                    'mean': mean,
                    'std': std,
                    'norm': norm
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for activation_stat, gradient_stat in zip(activation_stats, gradient_stats):
            if (activation_stat['dead_fraction'] > 0.5):
                return 'dead_neurons'
            elif (gradient_stat['norm'] > 1000):
                return 'exploding_gradients'
            elif gradient_stat['norm'] < 1e-5:
                return 'vanishing_gradients'
            elif (activation_stat['std'] < 0.1):
                return 'vanishing_gradients'
            elif (activation_stat['std'] > 10.0):
                return 'exploding_gradients'

        return 'healthy'















