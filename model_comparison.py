from scipy.stats import ttest_rel, wilcoxon
import numpy as np


def compare_models(model_predictions, true_values, alpha=0.05, n_bootstrap=1000):
    """
    Расширенное сравнение моделей с доверительными интервалами и статистическими тестами

    Параметры:
    ----------
    model_predictions : dict
        Словарь с предсказаниями моделей в формате {'имя_модели': [предсказания]}
    true_values : array-like
        Истинные значения целевой переменной
    alpha : float, optional
        Уровень значимости для статистических тестов (по умолчанию 0.05)
    n_bootstrap : int, optional
        Количество итераций бутстрепа для расчета доверительных интервалов (по умолчанию 1000)

    Возвращает:
    -----------
    dict
        Словарь с полной статистикой по сравнению моделей
    """

    # Рассчитываем абсолютные ошибки для каждой модели
    errors = {name: np.abs(pred - true_values)
              for name, pred in model_predictions.items()}

    # Основные статистики
    stats = {}
    for name, err in errors.items():
        stats[name] = {
            'средняя_ошибка': np.mean(err),
            'стандартное_отклонение': np.std(err),
            'медианная_ошибка': np.median(err),
            'максимальная_ошибка': np.max(err),
            'минимальная_ошибка': np.min(err),
            'квантиль_25': np.percentile(err, 25),
            'квантиль_75': np.percentile(err, 75)
        }

    # Бутстреп-доверительные интервалы
    print("\nРасчет бутстреп-доверительных интервалов...")
    for name, err in errors.items():
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(err, size=len(err), replace=True)
            bootstrap_means.append(np.mean(sample))

        stats[name]['нижняя_граница_95%_ДИ'] = np.percentile(bootstrap_means, 2.5)
        stats[name]['верхняя_граница_95%_ДИ'] = np.percentile(bootstrap_means, 97.5)

    # Вывод основных статистик
    print("\n=== ОСНОВНЫЕ СТАТИСТИКИ МОДЕЛЕЙ ===")
    for name, stat in stats.items():
        print(f"\nМодель: {name}")
        print("--------------------------------")
        print(f"Средняя ошибка: {stat['средняя_ошибка']:.4f}")
        print(
            f"95% доверительный интервал: [{stat['нижняя_граница_95%_ДИ']:.4f}, {stat['верхняя_граница_95%_ДИ']:.4f}]")
        print(f"Стандартное отклонение: {stat['стандартное_отклонение']:.4f}")
        print(f"Медианная ошибка: {stat['медианная_ошибка']:.4f}")
        print(f"Межквартильный размах: {stat['квантиль_25']:.4f} - {stat['квантиль_75']:.4f}")
        print(f"Минимальная ошибка: {stat['минимальная_ошибка']:.4f}")
        print(f"Максимальная ошибка: {stat['максимальная_ошибка']:.4f}")

    # Статистические тесты (если есть хотя бы 2 модели)
    models = list(model_predictions.keys())
    if len(models) >= 2:
        print("\n=== СТАТИСТИЧЕСКИЕ СРАВНЕНИЯ МОДЕЛЕЙ ===")
        print(f"Уровень значимости alpha = {alpha}")

        # Парные t-тесты
        print("\nПарные t-тесты Стьюдента для зависимых выборок:")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                t_stat, p_value = ttest_rel(errors[models[i]], errors[models[j]])
                print(f"\nСравнение {models[i]} vs {models[j]}:")
                print(f"t-статистика = {t_stat:.4f}")
                print(f"p-value = {p_value:.6f}")

                if p_value < alpha:
                    better = models[i] if stats[models[i]]['средняя_ошибка'] < stats[models[j]]['средняя_ошибка'] else \
                    models[j]
                    print(f"СТАТИСТИЧЕСКИ ЗНАЧИМО: {better} работает достоверно лучше (p-value < {alpha})")
                else:
                    print("Различия не достигли уровня статистической значимости")

        # Непараметрические тесты Вилкоксона
        print("\nНепараметрические тесты Вилкоксона для зависимых выборок:")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                if len(errors[models[i]]) == len(errors[models[j]]):
                    w_stat, p_value = wilcoxon(errors[models[i]], errors[models[j]])
                    print(f"\nСравнение {models[i]} vs {models[j]}:")
                    print(f"W-статистика = {w_stat:.4f}")
                    print(f"p-value = {p_value:.6f}")

                    if p_value < alpha:
                        better = models[i] if stats[models[i]]['медианная_ошибка'] < stats[models[j]][
                            'медианная_ошибка'] else models[j]
                        print(f"СТАТИСТИЧЕСКИ ЗНАЧИМО: {better} работает достоверно лучше (p-value < {alpha})")
                    else:
                        print("Различия не достигли уровня статистической значимости")



    return stats
