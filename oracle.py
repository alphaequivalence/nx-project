""" Original implementation from
    https://github.com/AlexImmer/FWDL/blob/master/oracles.py
"""

import numpy as np


def linear_minimization_oracle(grad, mu):
    shape = grad.shape
    grad = grad.reshape(-1)
    s = np.zeros(grad.shape)
    coord = np.argmax(np.abs(grad))
    s[coord] = mu * np.sign(grad[coord])
    return - s.reshape(*shape)
