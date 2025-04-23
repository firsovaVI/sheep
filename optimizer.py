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
            best_idx = np.argmin(fitness)
            best_params = transform_u_to_q(population[best_idx], self.param_bounds)

            # Сохраняем параметры с именами
            param_dict = {f"param_{i}": float(best_params[i]) for i in range(len(best_params))}

            data = {
                "iteration": iteration,
                "best_params": param_dict,
                "best_fitness": float(fitness[best_idx]),
                "method": "DEEP" if self.current_method == 0 else "Bandit"
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Error saving iteration data: {e}")

    def optimize(self, max_iterations=100):
        try:
            population = initialize_population(50, self.param_bounds)
            fitness = evaluate_population(population, self.objective_function)

            # Warm-up phase
            for iteration in range(16):
                new_population = recombine(population, 0.5, 0.7)
                new_fitness = evaluate_population(new_population, self.objective_function)
                population, fitness = select(population, new_population, fitness, new_fitness)

                # Record history
                self.f_best_history.append(np.min(fitness))
                self.variance_history.append(np.var(population))

                self._save_iteration_data(iteration, population, fitness)

            # Main optimization loop
            for iteration in range(16, max_iterations):
                # Predict next state
                self.current_method = self.hmm.predict_next_state(self.current_method)

                # Run optimization step
                if self.current_method == 0:  # DEEP
                    new_population = recombine(population, 0.5, 0.7)
                else:  # Bandit
                    bandit = BernoulliBandit(n=len(population))
                    strategy_idx = EpsilonGreedy(bandit).run_one_step()
                    new_population = population.copy()
                    mask = np.random.rand(*new_population.shape) < 0.2
                    new_population += mask * np.random.normal(0, 0.2, size=new_population.shape)

                new_fitness = evaluate_population(new_population, self.objective_function)
                population, fitness = select(population, new_population, fitness, new_fitness)

                # Update history
                self.f_best_history.append(np.min(fitness))
                self.variance_history.append(np.var(population))

                # Update HMM with new observations
                self.hmm.update_with_optimizer_data(
                    np.array(self.f_best_history[-16:]),
                    np.array(self.variance_history[-16:])
                )

                self._save_iteration_data(iteration, population, fitness)

                if iteration % 10 == 0:
                    best_idx = np.argmin(fitness)
                    print(f"Iter {iteration}: Method={'DEEP' if self.current_method == 0 else 'Bandit'}, "
                          f"Fitness={fitness[best_idx]:.4f}")

            best_idx = np.argmin(fitness)
            return transform_u_to_q(population[best_idx], self.param_bounds), fitness[best_idx]

        except Exception as e:
            print(f"Error in optimize(): {e}")
            return np.zeros(len(self.param_bounds)), float('inf')
