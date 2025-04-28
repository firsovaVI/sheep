from __future__ import division
import time
import numpy as np

class Bandit(object):
    def generate_reward(self, i):
        raise NotImplementedError

class BernoulliBandit(Bandit):
    def __init__(self, n, probas=None):
        assert probas is None or len(probas) == n
        self.n = n
        if probas is None:
            np.random.seed(int(time.time()))
            self.probas = [np.random.random() for _ in range(self.n)]
        else:
            self.probas = probas
        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        return 1 if np.random.random() < self.probas[i] else 0

    def generate_reward(self, i):
        return 1 if np.random.random() < self.probas[i] else 0
