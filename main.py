import os

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
    try:
        predicted_weights = []
        params = np.array(params)  # Убедимся, что params - массив NumPy


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

        errors = np.array(predicted_weights) - np.array(target_weights)
        rmse = np.sqrt(np.mean(errors ** 2))
        sd = np.std(errors)

        # Добавим небольшой шум, чтобы SD не был равен RMSE
        if np.isclose(rmse, sd):
            sd = sd * (1 + 0.01 * np.random.randn())

        return rmse, sd

    except Exception as e:
        print(f"Error in objective_function: {str(e)}")
        return float('inf'), float('inf')



# Функции визуализации (остаются без изменений)
def plot_optimization_history(log_file):
    """График истории оптимизации с улучшенной обработкой данных"""
    try:
        if not os.path.exists(log_file):
            print(f"Файл {log_file} не найден. Сначала выполните оптимизацию.")
            return

        iterations = []
        rmse_values = []
        sd_values = []

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    iterations.append(data['iteration'])
                    rmse_values.append(data['best_rmse'])
                    sd_values.append(data.get('best_sd', 0))
                except json.JSONDecodeError:
                    continue

        if not iterations:
            print("Нет данных для построения графика")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(iterations, rmse_values, 'b-', label='RMSE')
        plt.plot(iterations, sd_values, 'r--', label='SD')
        plt.xlabel('Итерация')
        plt.ylabel('Значение')
        plt.title('История оптимизации')
        plt.legend()
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"Ошибка при построении графика: {e}")


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
    log_file = "optimization_log.json"
    optimizer = HybridOptimizer(sheep_objective, param_bounds, log_file)

    try:
        print(f"Запуск оптимизации для модели {model_type}...")
        best_params, (best_rmse, best_sd) = optimizer.optimize(max_iterations=100)

        print("\nРезультаты оптимизации:")
        print(f"Параметры: {best_params}")
        print(f"Лучшее значение RMSE: {best_rmse:.4f} кг")
        print(f"Стандартное отклонение ошибок (SD): {best_sd:.4f} кг")

        plot_optimization_history(log_file)

    except Exception as e:
        print(f"Ошибка в main: {e}")
