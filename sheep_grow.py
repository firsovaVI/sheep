import numpy as np
import matplotlib.pyplot as plt
import json
from optimizer import HybridOptimizer
from matplotlib.gridspec import GridSpec

# Данные чистых пород из Таблицы 2 (коды A-F)
pure_breeds_data = {
    "A": {"name": "Santa Inês", "sex": {"m": [3.50, 13.50, 22.70, 24.80, 24.90],
                                        "f": [3.50, 13.50, 22.70, 24.80, 24.90]}},
    "B": {"name": "Hemşin", "sex": {"m": [4.00, 19.00, 38.00, 39.60, 42.00],
                                    "f": [4.00, 19.00, 36.00, 37.20, 46.00]}},
    "C": {"name": "Mecheri", "sex": {"m": [2.44, 10.09, 13.10, 16.18, 19.42],
                                     "f": [2.29, 9.09, 12.14, 14.75, 17.10]}},
    "D": {"name": "Deccani", "sex": {"m": [2.50, 12.00, 24.50, 25.00, 27.50],
                                     "f": [2.50, 11.00, 20.00, 21.00, 22.00]}},
    "E": {"name": "Konya Merino", "sex": {"m": [4.00, 17.34, 35.08, 37.83, 56.92],
                                          "f": [4.00, 14.94, 30.00, 32.41, 46.25]}},
    "F": {"name": "Lohi", "sex": {"m": [2.75, 11.70, 19.00, 21.00, 28.45],
                                  "f": [2.60, 11.50, 18.30, 20.70, 26.60]}}
}

# Данные гибридов из Таблицы 4 (средние значения)
hybrid_data = {
    "Am+Bf": {"age": [1, 60, 180, 210, 360], "weight": [3.75, 16.25, 29.35, 31.00, 35.45]},
    "Am+Cf": {"age": [1, 60, 180, 210, 360], "weight": [2.90, 11.30, 17.42, 19.78, 21.00]},
    "Am+Df": {"age": [1, 60, 180, 210, 360], "weight": [3.00, 12.25, 21.35, 22.90, 23.45]},
    "Am+Ef": {"age": [1, 60, 180, 210, 360], "weight": [3.75, 14.22, 26.35, 28.61, 35.58]},
    "Am+Ff": {"age": [1, 60, 180, 210, 360], "weight": [3.05, 12.50, 20.50, 22.75, 25.75]},
    "Bm+Af": {"age": [1, 60, 180, 210, 360], "weight": [3.75, 16.25, 30.35, 32.20, 33.45]},
    "Bm+Cf": {"age": [1, 60, 180, 210, 360], "weight": [3.15, 14.05, 25.07, 27.18, 29.55]},
    "Bm+Df": {"age": [1, 60, 180, 210, 360], "weight": [3.25, 15.00, 29.00, 30.30, 32.00]},
    "Bm+Ef": {"age": [1, 60, 180, 210, 360], "weight": [4.00, 16.97, 34.00, 36.01, 44.13]},
    "Bm+Ff": {"age": [1, 60, 180, 210, 360], "weight": [3.30, 15.25, 28.15, 30.15, 34.30]},
    "Cm+Af": {"age": [1, 60, 180, 210, 360], "weight": [2.97, 11.80, 17.90, 20.49, 22.16]},
    "Cm+Bf": {"age": [1, 60, 180, 210, 360], "weight": [3.22, 14.55, 24.55, 26.69, 32.71]},
    "Cm+Df": {"age": [1, 60, 180, 210, 360], "weight": [2.47, 10.55, 16.55, 18.59, 20.71]},
    "Cm+Ef": {"age": [1, 60, 180, 210, 360], "weight": [3.22, 12.52, 21.55, 24.30, 32.84]},
    "Cm+Ff": {"age": [1, 60, 180, 210, 360], "weight": [2.52, 10.80, 15.70, 18.44, 23.01]},
    "Dm+Af": {"age": [1, 60, 180, 210, 360], "weight": [3.00, 12.75, 23.60, 24.90, 26.20]},
    "Dm+Bf": {"age": [1, 60, 180, 210, 360], "weight": [3.25, 15.50, 30.25, 31.10, 36.75]},
    "Dm+Cf": {"age": [1, 60, 180, 210, 360], "weight": [2.40, 10.55, 18.32, 19.88, 22.30]},
    "Dm+Ef": {"age": [1, 60, 180, 210, 360], "weight": [3.25, 13.47, 27.25, 28.71, 36.88]},
    "Dm+Ff": {"age": [1, 60, 180, 210, 360], "weight": [2.55, 11.75, 21.40, 22.85, 27.05]},
    "Em+Af": {"age": [1, 60, 180, 210, 360], "weight": [3.75, 15.42, 28.89, 31.32, 40.91]},
    "Em+Bf": {"age": [1, 60, 180, 210, 360], "weight": [4.00, 18.17, 35.54, 37.52, 51.46]},
    "Em+Cf": {"age": [1, 60, 180, 210, 360], "weight": [3.15, 13.22, 23.61, 26.29, 37.01]},
    "Em+Df": {"age": [1, 60, 180, 210, 360], "weight": [3.25, 14.17, 27.54, 29.42, 39.46]},
    "Em+Ff": {"age": [1, 60, 180, 210, 360], "weight": [3.30, 14.42, 26.69, 29.27, 41.76]},
    "Fm+Af": {"age": [1, 60, 180, 210, 360], "weight": [3.13, 12.60, 20.85, 22.90, 26.68]},
    "Fm+Bf": {"age": [1, 60, 180, 210, 360], "weight": [3.38, 15.35, 27.50, 29.10, 37.23]},
    "Fm+Cf": {"age": [1, 60, 180, 210, 360], "weight": [2.52, 10.40, 15.57, 17.88, 22.78]},
    "Fm+Df": {"age": [1, 60, 180, 210, 360], "weight": [2.63, 11.35, 19.50, 21.00, 25.23]},
    "Fm+Ef": {"age": [1, 60, 180, 210, 360], "weight": [3.38, 13.32, 24.50, 26.71, 37.35]}
}

# Возрастные точки для всех данных (дни)
age_points = [1, 60, 180, 210, 360]

# Допустимые диапазоны параметров из Таблицы 5
model_parameter_ranges = {
    "brody": {"A": (31.2, 124.8), "B": (0.4, 1.6), "k": (0.0012, 0.0048)},
    "gompertz": {"A": (14.064, 56.256), "B": (1.4014, 5.6056), "k": (0.00312, 0.01248)},
    "logistic": {"A": (14.064, 56.256), "B": (1.4014, 5.6056), "k": (0.00312, 0.01248)},
    "negative_exponential": {"A": (14.064, 56.256), "k": (0.00312, 0.01248)},
    "richards": {"A": (32.8, 131.2), "B": (0.392, 1.568), "k": (0.0008, 0.0032), "m": (0.3032, 1.2128)},
    "von_bertalanffy": {"A": (30, 120), "B": (0.2, 0.8), "k": (0.0016, 0.0064)}
}


# Модели роста из статьи
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


def calculate_hybrid_curve(male_breed, female_breed):
    """Вычисляет гибридную кривую как среднее между породами"""
    male_weights = pure_breeds_data[male_breed]["sex"]["m"]
    female_weights = pure_breeds_data[female_breed]["sex"]["f"]
    return [(m + f) / 2 for m, f in zip(male_weights, female_weights)]


# Целевая функция для оптимизации
def objective_function(params, target_weights, model_type="brody"):
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

    # Рассчитываем RMSE
    return np.sqrt(np.mean((np.array(predicted_weights) - target_weights) ** 2))


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


def plot_growth_curve_comparison(best_params, target_weights, model_type="brody"):
    """Сравнение реальных и предсказанных кривых роста"""
    predicted_weights = []
    extended_ages = np.linspace(1, 360, 100)  # Для плавных кривых

    for age in extended_ages:
        if model_type == "brody":
            predicted = brody_model(best_params, age)
        elif model_type == "gompertz":
            predicted = gompertz_model(best_params, age)
        elif model_type == "logistic":
            predicted = logistic_model(best_params, age)
        elif model_type == "negative_exponential":
            predicted = negative_exponential_model(best_params, age)
        elif model_type == "richards":
            predicted = richards_model(best_params, age)
        elif model_type == "von_bertalanffy":
            predicted = von_bertalanffy_model(best_params, age)

        predicted_weights.append(predicted)

    plt.figure(figsize=(10, 6))
    plt.plot(age_points, target_weights, 'bo', label="Реальные данные")
    plt.plot(extended_ages, predicted_weights, 'r-', label=f"Модель {model_type}")
    plt.xlabel("Возраст (дни)")
    plt.ylabel("Вес (кг)")
    plt.title("Сравнение кривых роста")
    plt.legend()
    plt.grid(True)
    plt.savefig('growth_curve_comparison.png', dpi=300)
    plt.show()


def plot_parameter_evolution(log_file, param_names):
    """График изменения параметров"""
    with open(log_file) as f:
        data = [json.loads(line) for line in f]

    iterations = [entry['iteration'] for entry in data]
    params_history = {name: [] for name in param_names}

    for entry in data:
        for name in param_names:
            params_history[name].append(entry['best_params'][name])

    plt.figure(figsize=(12, 6))
    for name in param_names:
        plt.plot(iterations, params_history[name], label=name)

    plt.xlabel('Итерация')
    plt.ylabel('Значение параметра')
    plt.title('Эволюция параметров модели')
    plt.legend()
    plt.grid(True)
    plt.savefig('parameter_evolution.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Конфигурация оптимизации
    model_type = "brody"  # Можно выбрать любую модель из model_parameter_ranges
    hybrid_name = "Am+Bf"  # Или любой другой гибрид из hybrid_data

    # Получаем целевые веса (либо из гибридных данных, либо вычисляем)
    if hybrid_name in hybrid_data:
        target_weights = hybrid_data[hybrid_name]
    else:
        # Если гибрида нет в таблице 4, вычисляем его как среднее
        male_breed = hybrid_name.split('+')[0][0]
        female_breed = hybrid_name.split('+')[1][0]
        target_weights = calculate_hybrid_curve(male_breed, female_breed)

    # Получаем границы параметров для выбранной модели
    param_bounds = list(model_parameter_ranges[model_type].values())
    param_names = list(model_parameter_ranges[model_type].keys())


    # Создаем функцию цели для конкретного гибрида и модели
    def hybrid_objective(params):
        return objective_function(params, target_weights, model_type)


    # Настройка оптимизатора
    log_file = "optimization_log.json"
    optimizer = HybridOptimizer(hybrid_objective, param_bounds, log_file)

    try:
        print(f"Запуск оптимизации для модели {model_type} и гибрида {hybrid_name}...")
        best_params, best_fitness = optimizer.optimize(max_iterations=100)

        print("\nРезультаты оптимизации:")
        for name, value in zip(param_names, best_params):
            print(f"{name} = {value:.4f}")
        print(f"Лучшее значение RMSE: {best_fitness:.4f}")

        # Визуализация результатов
        plot_optimization_history(log_file)
        plot_growth_curve_comparison(best_params, target_weights, model_type)
        plot_parameter_evolution(log_file, param_names)

    except Exception as e:
        print(f"Ошибка в main: {e}")
