import numpy as np
from deep_core import initialize_population, recombine, evaluate_population, select, transform_u_to_q  # Убрана точка перед deep_core


def adapt_parameters(F, crossover_prob, population, fitness):
    pass


def deep_algorithm(objective_function, param_bounds, pop_size=50, max_generations=100, F=0.5, crossover_prob=0.7):

    """
    Основной алгоритм DEEP.

    :param objective_function: Целевая функция.
    :param param_bounds: Границы параметров.
    :param pop_size: Размер популяции.
    :param max_generations: Максимальное количество поколений.
    :param F: Коэффициент масштабирования.
    :param crossover_prob: Вероятность кроссовера.
    :return: Лучший индивид и его значение целевой функции.
    """
    # Инициализация популяции
    population = initialize_population(pop_size, param_bounds)
    fitness = evaluate_population(population, objective_function)

    for generation in range(max_generations):
        # Рекомбинация
        new_population = recombine(population, F, crossover_prob)

        # Оценка новой популяции
        new_fitness = evaluate_population(new_population, objective_function)

        # Селекция
        population, fitness = select(population, new_population, fitness, new_fitness)

        # Адаптация параметров
        F, crossover_prob = adapt_parameters(F, crossover_prob, population, fitness)

        # Вывод информации о текущем поколении
        best_fitness = np.min(fitness)
        print(f"Generation {generation + 1}, Best Fitness: {best_fitness}")

    # Возвращаем лучший индивид
    best_index = np.argmin(fitness)
    return population[best_index], fitness[best_index]
