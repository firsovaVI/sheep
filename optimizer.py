import numpy as np
from deep_core import *
import json
import os


class HybridOptimizer:
    def __init__(self, objective_function, param_bounds, log_file="optimization_log.json"):
        self.param_bounds = param_bounds
        self.objective_function = lambda x: objective_function(transform_u_to_q(x, param_bounds))
        self.log_file = log_file
        self.reset_log()

    def reset_log(self):
        """Очистка файла журнала"""
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _save_iteration_data(self, iteration, population, fitness):
        """Сохранение данных итерации"""
        try:
            best_idx = np.argmin(fitness[:, 0])
            best_params = transform_u_to_q(population[best_idx], self.param_bounds)

            data = {
                "iteration": iteration,
                "best_params": {f"param_{i}": float(best_params[i]) for i in range(len(best_params))},
                "best_rmse": float(fitness[best_idx, 0]),
                "best_sd": float(fitness[best_idx, 1])
            }

            with open(self.log_file, 'a') as f:
                f.write(json.dumps(data) + '\n')
        except Exception as e:
            print(f"Ошибка при сохранении данных итерации: {e}")

    def optimize(self, max_iterations=100):
        """Основной метод оптимизации"""
        try:
            population = initialize_population(50, self.param_bounds)
            fitness = np.zeros((len(population), 2))

            # Инициализация fitness
            for i, ind in enumerate(population):
                rmse, sd = self.objective_function(ind)
                fitness[i] = [rmse, sd]

            self._save_iteration_data(0, population, fitness)

            for iteration in range(1, max_iterations + 1):
                # Адаптация параметров
                F, crossover_prob = adapt_parameters(0.5, 0.7, population, fitness[:, 0])

                # Рекомбинация
                new_population = recombine(population, F, crossover_prob)
                new_fitness = np.zeros((len(new_population), 2))

                # Оценка новых особей
                for i, ind in enumerate(new_population):
                    rmse, sd = self.objective_function(ind)
                    new_fitness[i] = [rmse, sd]

                # Селекция
                for i in range(len(population)):
                    if new_fitness[i, 0] < fitness[i, 0]:
                        population[i] = new_population[i]
                        fitness[i] = new_fitness[i]

                # Логирование
                self._save_iteration_data(iteration, population, fitness)

                if iteration % 10 == 0:
                    best_idx = np.argmin(fitness[:, 0])
                    print(f"Итерация {iteration}: RMSE={fitness[best_idx, 0]:.4f}")

            best_idx = np.argmin(fitness[:, 0])
            return (transform_u_to_q(population[best_idx], self.param_bounds),
                    (fitness[best_idx, 0], fitness[best_idx, 1]))

        except Exception as e:
            print(f"Ошибка в optimize(): {str(e)}")
            return np.zeros(len(self.param_bounds)), (float('inf'), 0.0)
