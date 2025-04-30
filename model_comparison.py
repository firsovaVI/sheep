from scipy.stats import ttest_rel, mannwhitneyu, tukey_hsd
import numpy as np


def compare_models(model_predictions, true_values, alpha=0.05):
    """
    Сравнение моделей с использованием статистических тестов
    """
    errors = {name: np.abs(pred - true_values)
              for name, pred in model_predictions.items()}

    # Попарные t-тесты
    print("\nРезультаты парных t-тестов:")
    models = list(model_predictions.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            t_stat, p_val = ttest_rel(errors[models[i]], errors[models[j]])
            print(f"{models[i]} vs {models[j]}: p-value={p_val:.4f}")
            if p_val < alpha:
                better = models[i] if np.mean(errors[models[i]]) < np.mean(errors[models[j]]) else models[j]
                print(f"  {better} показал значительно лучшие результаты")

    # Непараметрический тест
    print("\nРезультаты теста Манна-Уитни:")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            u_stat, p_val = mannwhitneyu(errors[models[i]], errors[models[j]])
            print(f"{models[i]} vs {models[j]}: p-value={p_val:.4f}")

    # Тест Тьюки для множественных сравнений
    if len(models) > 2:
        print("\nРезультаты теста Тьюки:")
        res = tukey_hsd(*[errors[m] for m in models])
        print(res)
