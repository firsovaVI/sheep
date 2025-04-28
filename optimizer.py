import numpy as np
from deep_core import *
from bandit import BernoulliBandit
from solver import EpsilonGreedy
from hmm import HiddenMarkovModel
import json
import os


class HybridOptimizer:
    def __init__(self, objective_function, param_bounds, log_file="optimization_log.json"):
        self.param_bounds = param_bounds
        self.objective_function = lambda x: objective_function(transform_u_to_q(x, param_bounds))
        self.hmm = HiddenMarkovModel(n_states=2, observation_levels=8)
        self.current_method = 0
        self.log_file = log_file
        self.history = []
        self.f_best_history = []
        self.variance_history = []

        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _save_iteration_data(self, iteration, population, fitness):
        try:
            best_idx = np.argmin(fitness[:, 0])  # Индекс по RMSE
            best_params = transform_u_to_q(population[best_idx], self.param_bounds)
            best_rmse = fitness[best_idx, 0]
            best_sd = fitness[best_idx, 1]

            data = {
                "iteration": iteration,
                "best_params": {f"param_{i}": float(best_params[i]) for i in range(len(best_params))},
                "best_rmse": float(best_rmse),
                "best_sd": float(best_sd),
                "method": "DEEP" if self.current_method == 0 else "Bandit"
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error saving iteration data: {e}")

    def optimize(self, max_iterations=100):
        try:
            population = initialize_population(50, self.param_bounds)
            fitness = np.zeros((len(population), 2))  # Теперь храним RMSE и SD

            # Инициализация fitness
            for i, ind in enumerate(population):
                result = self.objective_function(ind)
                if isinstance(result, tuple) and len(result) == 2:
                    rmse, sd = result
                else:
                    rmse = result if isinstance(result, (float, np.float64)) else float('inf')
                    sd = 0.0
                fitness[i] = [rmse, sd]

            # Сохраняем начальное состояние
            self._save_iteration_data(0, population, fitness)

            for iteration in range(1, max_iterations + 1):
                new_population = recombine(population, 0.5, 0.7)
                new_fitness = np.zeros((len(new_population), 2))

                for i, ind in enumerate(new_population):
                    result = self.objective_function(ind)
                    if isinstance(result, tuple) and len(result) == 2:
                        rmse, sd = result
                    else:
                        rmse = result if isinstance(result, (float, np.float64)) else float('inf')
                        sd = 0.0
                    new_fitness[i] = [rmse, sd]

                # Селекция
                for i in range(len(population)):
                    if new_fitness[i, 0] < fitness[i, 0]:
                        population[i] = new_population[i]
                        fitness[i] = new_fitness[i]

                # Сохраняем данные итерации
                self._save_iteration_data(iteration, population, fitness)

                if iteration % 10 == 0:
                    best_idx = np.argmin(fitness[:, 0])
                    print(f"Iter {iteration}: RMSE={fitness[best_idx, 0]:.4f}, SD={fitness[best_idx, 1]:.4f}")

            best_idx = np.argmin(fitness[:, 0])
            best_params = transform_u_to_q(population[best_idx], self.param_bounds)
            return best_params, (fitness[best_idx, 0], fitness[best_idx, 1])

        except Exception as e:
            print(f"Error in optimize(): {str(e)}")
            return np.zeros(len(self.param_bounds)), (float('inf'), 0.0)
