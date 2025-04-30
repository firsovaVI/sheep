import numpy as np
from deep_core import *
from bandit import BernoulliBandit
from solver import ThompsonSampling
import json
import os


class HybridOptimizer:
    def __init__(self, objective_function, param_bounds, log_file="optimization_log.json"):
        self.param_bounds = param_bounds
        self.objective_function = lambda x: objective_function(transform_u_to_q(x, param_bounds))
        self.log_file = log_file
        self.reset_log()

        # Инициализация Bandit для выбора между DEEP и случайным исследованием
        self.bandit = BernoulliBandit(n=2)  # 0 - DEEP, 1 - Bandit
        self.solver = ThompsonSampling(self.bandit)
        self.method_history = []

        # Статистика для адаптации
        self.last_improvement = [0, 0]  # Улучшения для каждого метода
        self.method_counts = [0, 0]

    def reset_log(self):
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _save_iteration_data(self, iteration, population, fitness, method):
        best_idx = np.argmin(fitness[:, 0])
        best_params = transform_u_to_q(population[best_idx], self.param_bounds)

        data = {
            "iteration": iteration,
            "method": method,
            "best_params": {f"param_{i}": float(best_params[i])
                            for i in range(len(best_params))},
            "best_rmse": float(fitness[best_idx, 0]),
            "best_sd": float(fitness[best_idx, 1])
        }

        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _bandit_step(self, population, fitness):
        # Случайное исследование пространства
        new_population = population.copy()
        random_idx = np.random.randint(0, len(population))

        # Генерация случайной мутации
        mutation = np.random.normal(0, 0.2, size=population.shape[1])
        new_population[random_idx] = np.clip(
            population[random_idx] + mutation,
            -1, 1  # Ограничения для U-пространства
        )

        # Оценка новой особи
        new_fitness = np.zeros(2)
        rmse, sd = self.objective_function(new_population[random_idx])
        new_fitness[0], new_fitness[1] = rmse, sd

        # Селекция
        if new_fitness[0] < fitness[random_idx, 0]:
            population[random_idx] = new_population[random_idx]
            fitness[random_idx] = new_fitness
            return 1  # Награда за улучшение
        return 0  # Нет награды

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

        return 1 if improved > 0 else 0

    def optimize(self, max_iterations=100):
        population = initialize_population(50, self.param_bounds)
        fitness = np.zeros((len(population), 2))

        # Инициализация fitness
        for i, ind in enumerate(population):
            rmse, sd = self.objective_function(ind)
            fitness[i] = [rmse, sd]

        self._save_iteration_data(0, population, fitness, method='INIT')

        for iteration in range(1, max_iterations + 1):
            # Выбор метода через Bandit
            method_idx = self.solver.run_one_step()
            method = 'DEEP' if method_idx == 0 else 'Bandit'
            self.method_counts[method_idx] += 1

            # Выполнение выбранного метода
            if method == 'DEEP':
                reward = self._deep_step(population, fitness)
            else:
                reward = self._bandit_step(population, fitness)

            # Обновление Bandit
            self.bandit.generate_reward(method_idx, reward)
            self.last_improvement[method_idx] += reward
            self.method_history.append(method)

            # Логирование
            if iteration % 10 == 0 or iteration == max_iterations:
                best_idx = np.argmin(fitness[:, 0])
                print(f"Iter {iteration}: {method} | RMSE={fitness[best_idx, 0]:.4f}")
                self._save_iteration_data(iteration, population, fitness, method)

        # Возврат лучшего решения
        best_idx = np.argmin(fitness[:, 0])
        return (transform_u_to_q(population[best_idx], self.param_bounds),
                (fitness[best_idx, 0], fitness[best_idx, 1]))
