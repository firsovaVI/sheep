from optimizer import HybridOptimizer
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.gridspec import GridSpec


# Модели роста
def brody_model(params, age):
    A, B, k = params
    return A * (1 - B * np.exp(-k * age))


def gompertz_model(params, age):
    A, B, k = params
    return A * np.exp(-B * np.exp(-k * age))


def logistic_model(params, age):
    A, B, k = params
    return A / (1 + B * np.exp(-k * age))


def negative_exponential_model(params, age):
    A, k = params
    return A * (1 - np.exp(-k * age))


def richards_model(params, age):
    A, B, k, m = params
    return A * (1 - B * np.exp(-k * age)) ** (1 / m)


def von_bertalanffy_model(params, age):
    A, B, k = params
    return A * (1 - B * np.exp(-k * age)) ** 3


# Целевая функция
def objective_function(params, target_weights, age_points, model_type="brody"):
    predicted_weights = []

    for age, target_weight in zip(age_points, target_weights):
        if model_type == "brody":
            predicted = brody_model(params, age)
        elif model_type == "gompertz":
            predicted = gompertz_model(params, age)
        elif model_type == "logistic":
            predicted = logistic_model(params, age)
        elif model_type == "negative_exponential":
            predicted = negative_exponential_model(params, age)
        elif model_type == "richards":
            predicted = richards_model(params, age)
        elif model_type == "von_bertalanffy":
            predicted = von_bertalanffy_model(params, age)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        predicted_weights.append(predicted)

    return np.sqrt(np.mean((np.array(predicted_weights) - target_weights) ** 2))


# Функции визуализации (остаются без изменений)
def plot_optimization_history(log_file):
    """График истории оптимизации"""
    with open(log_file) as f:
        data = [json.loads(line) for line in f]

    iterations = [entry['iteration'] for entry in data]
    fitness = [entry['best_fitness'] for entry in data]
    methods = [0 if entry['method'] == 'DEEP' else 1 for entry in data]

    plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    ax0.plot(iterations, fitness, 'b-', label='RMSE')
    ax0.set_ylabel('Ошибка (RMSE)')
    ax0.set_title('История оптимизации параметров модели')
    ax0.grid(True)
    ax0.legend()

    ax1 = plt.subplot(gs[1])
    colors = ['blue' if m == 0 else 'red' for m in methods]
    ax1.scatter(iterations, methods, c=colors, alpha=0.6, s=20)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['DEEP', 'Bandit'])
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Метод')
    ax1.grid(True)

    plt.tight_layout()
    plt.savefig('optimization_history.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Пример данных для оптимизации
    age_points = [1, 60, 180, 210, 360]  # дни
    target_weights = [3.5, 13.5, 22.7, 24.8, 24.9]  # кг

    # Выбор модели и параметров
    model_type = "brody"
    param_bounds = [(31.2, 124.8), (0.4, 1.6), (0.0012, 0.0048)]  # для модели Brody


    # Создаем функцию цели
    def sheep_objective(params):
        return objective_function(params, target_weights, age_points, model_type)


    # Настройка оптимизатора
    log_file = "sheep_optimization_log.json"
    optimizer = HybridOptimizer(sheep_objective, param_bounds, log_file)

    try:
        print(f"Запуск оптимизации для модели {model_type}...")
        best_params, best_fitness = optimizer.optimize(max_iterations=100)

        print("\nРезультаты оптимизации:")
        print(f"Параметры: {best_params}")
        print(f"Лучшее значение RMSE: {best_fitness:.4f}")

        plot_optimization_history(log_file)

    except Exception as e:
        print(f"Ошибка в main: {e}")
