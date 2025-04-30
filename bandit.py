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

    def generate_reward(self, i, reward=None):
        """Измененный метод для поддержки внешних вознаграждений"""
        if reward is None:
            return 1 if np.random.random() < self.probas[i] else 0
        else:
            # Обновляем вероятность при получении внешнего вознаграждения
            self.probas[i] = (self.probas[i] + reward) / 2
            return reward
