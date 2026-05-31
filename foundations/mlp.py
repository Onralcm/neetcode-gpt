import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        num_hidden_layers = len(weights)
        h = np.array([])
        for i in range(num_hidden_layers):
            if i == 0:
                h = np.maximum(0, x @ weights[i] + biases[i])
            elif i == num_hidden_layers - 1:
                h = h @ weights[i] + biases[i]
            else:
                h = np.maximum(0, h @ weights[i] + biases[i])

        return h

