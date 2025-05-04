import numpy as np
import json
import os


def generate_synthetic_data(mean_weights, std_devs=None, n_samples=100):
    """
    Генерация синтетических данных на основе средних значений и стандартных отклонений.
    Возвращает массив размером (n_samples, n_age_points).
    """
    if std_devs is None:
        std_devs = [m * 0.1 for m in mean_weights]  # 10% от среднего, если std не задан

    synthetic_data = []
    for mean, std in zip(mean_weights, std_devs):
        samples = np.random.normal(mean, std, n_samples)
        samples = np.clip(samples, 0, None)  # Обрезаем отрицательные значения
        synthetic_data.append(samples)

    return np.array(synthetic_data).T  # Транспонируем для формата (n_samples, n_age_points)


def save_synthetic_data_to_json(data, file_path="synthetic_data.json"):
    """
    Сохраняет синтетические данные в JSON-файл.

    Параметры:
        data (np.ndarray): Массив синтетических данных (n_samples, n_features).
        file_path (str): Путь к файлу для сохранения.
    """
    try:
        # Конвертируем numpy-массив в список (JSON не поддерживает numpy-типы)
        data_list = data.tolist()

        # Сохраняем в JSON
        with open(file_path, 'w') as f:
            json.dump({
                "synthetic_data": data_list,
                "metadata": {
                    "n_samples": data.shape[0],
                    "n_features": data.shape[1],
                    "description": "Synthetic sheep weight data (kg) for different ages."
                }
            }, f, indent=4)

        print(f"Данные успешно сохранены в {file_path}")
    except Exception as e:
        print(f"Ошибка при сохранении данных: {str(e)}")


def load_breed_data(breed_name, sex='m'):
    """
    Загрузка данных по породе из базы данных.
    """
    example_data = {
        'Santa_Ines': {
            'm': {'mean': [3.50, 13.50, 22.70, 24.80, 24.90],
                  'std': [0.35, 1.35, 2.27, 2.48, 2.49]},
            'f': {'mean': [3.50, 13.50, 22.70, 24.80, 24.90],
                  'std': [0.35, 1.35, 2.27, 2.48, 2.49]}
        }
    }
    return example_data.get(breed_name, {}).get(sex, {})
