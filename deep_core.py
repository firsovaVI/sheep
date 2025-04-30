import numpy as np


def transform_u_to_q(u, param_bounds):
    try:
        q = np.zeros_like(u)
        for i in range(len(u)):
            alpha = (param_bounds[i][1] + param_bounds[i][0]) / 2
            beta = (param_bounds[i][1] - param_bounds[i][0]) / 2
            q[i] = alpha + beta * np.tanh(u[i])
        return q
    except Exception as e:
        print(f"Ошибка в transform_u_to_q: {e}")
        return np.zeros_like(u)


def initialize_population(pop_size, param_bounds):
    try:
        num_params = len(param_bounds)
        return np.random.uniform(-1, 1, size=(pop_size, num_params))
    except Exception as e:
        print(f"Ошибка в initialize_population: {e}")
        return np.random.uniform(-1, 1, size=(pop_size, 3))  # fallback


def adapt_parameters(F, crossover_prob, population, fitness):
    """
    Улучшенная адаптация параметров
    """
    # Рассчитываем разнообразие популяции
    diversity = np.mean(np.std(population, axis=0))

    # Корректируем F на основе разнообразия
    if diversity < 0.1:  # Популяция сходится
        F *= 1.2  # Увеличиваем исследование
    else:
        F *= 0.9  # Фокусируемся на эксплуатации

    # Корректируем вероятность кроссовера
    avg_improvement = np.mean(np.maximum(0, fitness[:-1] - fitness[1:]))
    if avg_improvement < 0.01:  # Стагнация
        crossover_prob = min(0.9, crossover_prob * 1.1)
    else:
        crossover_prob = max(0.5, crossover_prob * 0.95)

    return np.clip(F, 0.1, 1.0), np.clip(crossover_prob, 0.5, 0.9)


def recombine(population, F, crossover_prob):
    try:
        pop_size, num_params = population.shape
        new_population = np.zeros_like(population)
        for i in range(pop_size):
            a, b, c = np.random.choice(pop_size, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            trial = np.where(np.random.rand(num_params) < crossover_prob, mutant, population[i])
            new_population[i] = trial
        return new_population
    except Exception as e:
        print(f"Error in recombine: {e}")
        return population.copy()

def evaluate_population(population, objective_function):
    try:
        return np.array([objective_function(ind) for ind in population])
    except Exception as e:
        print(f"Error in evaluate_population: {e}")
        return np.ones(len(population)) * float('inf')


def select(population, new_population, fitness, new_fitness):
    try:
        pop_size = population.shape[0]
        selected_population = np.zeros_like(population)
        selected_fitness = np.zeros_like(fitness)

        # Убедимся, что fitness и new_fitness - одномерные массивы
        if fitness.ndim > 1:
            fitness = fitness[:, 0]  # Берем только RMSE для сравнения
        if new_fitness.ndim > 1:
            new_fitness = new_fitness[:, 0]

        for i in range(pop_size):
            if new_fitness[i] < fitness[i]:
                selected_population[i] = new_population[i]
                selected_fitness[i] = new_fitness[i]
            else:
                selected_population[i] = population[i]
                selected_fitness[i] = fitness[i]
        return selected_population, selected_fitness
    except Exception as e:
        print(f"Error in select: {str(e)}")
        return population, fitness
