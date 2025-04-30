import numpy as np


def generate_synthetic_data(mean_weights, std_devs=None, n_samples=100):
    """
    Генерация синтетических данных на основе средних значений и стандартных отклонений
    """
    if std_devs is None:
        # Если стандартные отклонения не указаны, используем 10% от среднего
        std_devs = [m * 0.1 for m in mean_weights]

    synthetic_data = []
    for mean, std in zip(mean_weights, std_devs):
        # Генерация нормального распределения, обрезаем отрицательные значения
        samples = np.random.normal(mean, std, n_samples)
        samples = np.clip(samples, 0, None)
        synthetic_data.append(samples)

    return np.array(synthetic_data).T  # Возвращаем в формате (n_samples, n_age_points)


def load_breed_data(breed_name, sex='m'):
    """
    Загрузка данных по породе из базы данных
    """
    # Здесь должна быть реализация загрузки реальных данных
    # Временная заглушка с примером данных
    example_data = {
        'Santa_Ines': {
            'm': {'mean': [3.50, 13.50, 22.70, 24.80, 24.90],
                  'std': [0.35, 1.35, 2.27, 2.48, 2.49]},
            'f': {'mean': [3.50, 13.50, 22.70, 24.80, 24.90],
                  'std': [0.35, 1.35, 2.27, 2.48, 2.49]}
        }
    }
    return example_data.get(breed_name, {}).get(sex, {})
