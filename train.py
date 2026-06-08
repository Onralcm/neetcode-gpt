import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        loss = None
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            starts = torch.randint(len(data) - context_length, (batch_size, 1))
            indices = starts + torch.arange(context_length)
            X, Y = data[indices], data[indices + 1]
            logits = model(X)
            logits = logits.reshape(batch_size * context_length, -1)
            Y = Y.reshape(batch_size * context_length)
            loss = F.cross_entropy(logits, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
            
