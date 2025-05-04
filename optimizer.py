import numpy as np
from deep_core import *
from bandit import BernoulliBandit
from solver import ThompsonSampling
from hmm import HiddenMarkovModel
import json
import os


class HybridOptimizer:
    def __init__(self, objective_function, param_bounds, log_file="optimization_log.json"):
        self.param_bounds = param_bounds
        self.objective_function = lambda x: objective_function(transform_u_to_q(x, param_bounds))
        self.log_file = log_file
        self.reset_log()

        # Инициализация компонентов оптимизации
        self.hmm = HiddenMarkovModel(n_states=2, observation_levels=8)
        self.bandit = ThompsonSampling(BernoulliBandit(2))  # 0 - DEEP, 1 - Bandit
        
        # Трекинг состояния
        self.current_method = 0  # 0 - DEEP, 1 - Bandit
        self.f_best_history = []
        self.variance_history = []
        self.method_history = []

    def reset_log(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _save_iteration_data(self, iteration, population, fitness):
        best_idx = np.argmin(fitness[:, 0])
        best_params = transform_u_to_q(population[best_idx], self.param_bounds)

        data = {
            "iteration": iteration,
            "method": "DEEP" if self.current_method == 0 else "Bandit",
            "best_params": {f"param_{i}": float(best_params[i]) 
                          for i in range(len(best_params))},
            "best_rmse": float(fitness[best_idx, 0]),
            "best_sd": float(fitness[best_idx, 1]),
            "f_best": float(np.min(fitness[:, 0])),
            "population_variance": float(np.var(population))
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _deep_step(self, population, fitness):
        F, crossover_prob = adapt_parameters(0.5, 0.7, population, fitness[:, 0])
        new_population = recombine(population, F, crossover_prob)
        new_fitness = np.zeros((len(new_population), 2))

        for i, ind in enumerate(new_population):
            rmse, sd = self.objective_function(ind)
            new_fitness[i] = [rmse, sd]

        improved = 0
        for i in range(len(population)):
            if new_fitness[i, 0] < fitness[i, 0]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]
                improved += 1

        return improved > 0

    def _bandit_step(self, population, fitness):
        random_idx = np.random.randint(0, len(population))
        mutation = np.random.normal(0, 0.2, size=population.shape[1])
        new_ind = np.clip(population[random_idx] + mutation, -1, 1)
        
        rmse, sd = self.objective_function(new_ind)
        new_fitness = np.array([rmse, sd])

        if new_fitness[0] < fitness[random_idx, 0]:
            population[random_idx] = new_ind
            fitness[random_idx] = new_fitness
            return 1  # Награда за улучшение
        return 0  # Нет награды

    def optimize(self, max_iterations=100):
        population = initialize_population(50, self.param_bounds)
        fitness = np.zeros((len(population), 2))

        # Инициализация fitness
        for i, ind in enumerate(population):
            rmse, sd = self.objective_function(ind)
            fitness[i] = [rmse, sd]

        self._save_iteration_data(0, population, fitness)

        # Фаза разогрева (только DEEP)
        for iteration in range(1, 16):
            self._deep_step(population, fitness)
            self.f_best_history.append(np.min(fitness[:, 0]))
            self.variance_history.append(np.var(population))
            self._save_iteration_data(iteration, population, fitness)

        # Основной цикл оптимизации
        for iteration in range(16, max_iterations + 1):
            # Предсказание следующего состояния через HMM
            if len(self.f_best_history) >= 16:
                self.hmm.update_with_optimizer_data(
                    np.array(self.f_best_history[-16:]),
                    np.array(self.variance_history[-16:])
                )
                self.current_method = self.hmm.predict_next_state(self.current_method)
            
            # Выбор и выполнение метода оптимизации
            if self.current_method == 0:  # DEEP
                reward = self._deep_step(population, fitness)
            else:  # Bandit
                reward = self._bandit_step(population, fitness)

            # Обновление Bandit
            self.bandit.run_one_step()  # Thompson Sampling делает выбор автоматически
            self.bandit.bandit.generate_reward(self.current_method, reward)

            # Обновление истории
            self.f_best_history.append(np.min(fitness[:, 0]))
            self.variance_history.append(np.var(population))
            self.method_history.append(self.current_method)

            # Логирование
            if iteration % 10 == 0 or iteration == max_iterations:
                best_idx = np.argmin(fitness[:, 0])
                print(f"Iter {iteration}: Method={'DEEP' if self.current_method == 0 else 'Bandit'} | "
                      f"RMSE={fitness[best_idx, 0]:.4f} | "
                      f"Best params={transform_u_to_q(population[best_idx], self.param_bounds)}")
            
            self._save_iteration_data(iteration, population, fitness)

        # Возвращаем лучшее решение
        best_idx = np.argmin(fitness[:, 0])
        return (transform_u_to_q(population[best_idx], self.param_bounds),
                (fitness[best_idx, 0], fitness[best_idx, 1]))
