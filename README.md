# sheep
![image](https://github.com/user-attachments/assets/c56f425c-8da2-4ae1-805c-a4a0835ec4b3)
![image](https://github.com/user-attachments/assets/8279a1a3-e518-46f3-aeb3-fb0b9c6061ed)
![image](https://github.com/user-attachments/assets/78efce6f-4c3e-46be-96fb-732627c405c3)
![image](https://github.com/user-attachments/assets/2646b321-9662-46a7-8b2f-494228eebca1)
Данные успешно сохранены в synthetic_santa_ines_m.json
Loading data...

## === Optimizing brody model ===
Iter 10: Method=DEEP | RMSE=2.2531 | Best params=[2.39312582e+01 8.59723952e-01 1.21170009e-02]
Iter 20: Method=DEEP | RMSE=1.9540 | Best params=[2.65510129e+01 9.14762507e-01 1.06639514e-02]
Iter 30: Method=DEEP | RMSE=1.9186 | Best params=[2.61596187e+01 8.93077275e-01 1.10177618e-02]
Iter 40: Method=DEEP | RMSE=1.9143 | Best params=[2.61896659e+01 8.80704338e-01 1.05665061e-02]
Iter 50: Method=DEEP | RMSE=1.9139 | Best params=[2.62508991e+01 8.90373915e-01 1.06147126e-02]

### Results for brody:
Parameters: [2.62508991e+01 8.90373915e-01 1.06147126e-02]
RMSE: 1.9139 kg
Standard deviation: 1.7078 kg

## === Optimizing gompertz model ===
Iter 10: Method=Bandit | RMSE=1.8618 | Best params=[2.53849218e+01 2.06679905e+00 1.76813088e-02]
Iter 20: Method=DEEP | RMSE=1.8618 | Best params=[2.53849218e+01 2.06679905e+00 1.76813088e-02]
Iter 30: Method=Bandit | RMSE=1.8250 | Best params=[2.49422982e+01 2.04759432e+00 2.00704218e-02]
Iter 40: Method=Bandit | RMSE=1.8198 | Best params=[2.49497784e+01 2.01760276e+00 1.94151638e-02]
Iter 50: Method=Bandit | RMSE=1.8198 | Best params=[2.49497784e+01 2.01760276e+00 1.94151638e-02]

### Results for gompertz:
Parameters: [2.49497784e+01 2.01760276e+00 1.94151638e-02]
RMSE: 1.8198 kg
Standard deviation: 1.6046 kg

## === Optimizing logistic model ===
Iter 10: Method=Bandit | RMSE=2.1819 | Best params=[25.40964651  4.52318831  0.03065211]
Iter 20: Method=Bandit | RMSE=1.9434 | Best params=[24.25553721  4.37468552  0.02893379]
Iter 30: Method=DEEP | RMSE=1.8925 | Best params=[24.28250424  4.77887001  0.0284349 ]
Iter 40: Method=Bandit | RMSE=1.8774 | Best params=[24.43661167  4.93019724  0.02875419]
Iter 50: Method=DEEP | RMSE=1.8774 | Best params=[24.43661167  4.93019724  0.02875419]

### Results for logistic:
Parameters: [24.43661167  4.93019724  0.02875419]
RMSE: 1.8774 kg
Standard deviation: 1.6659 kg

## === Model Comparison ===

Расчет бутстреп-доверительных интервалов...

## === ОСНОВНЫЕ СТАТИСТИКИ МОДЕЛЕЙ ===

### Модель: brody
--------------------------------
Средняя ошибка: 7.6026
95% доверительный интервал: [5.2662, 11.0678]
Стандартное отклонение: 3.5145
Медианная ошибка: 6.8315
Межквартильный размах: 5.1634 - 7.0597
Минимальная ошибка: 4.5865
Максимальная ошибка: 14.3719

### Модель: gompertz
--------------------------------
Средняя ошибка: 7.6047
95% доверительный интервал: [5.2683, 11.0200]
Стандартное отклонение: 3.5101
Медианная ошибка: 6.8419
Межквартильный размах: 5.1738 - 7.0702
Минимальная ошибка: 4.5761
Максимальная ошибка: 14.3615

### Модель: logistic
--------------------------------
Средняя ошибка: 7.5902
95% доверительный интервал: [5.1633, 11.0346]
Стандартное отклонение: 3.5416
Медианная ошибка: 6.7692
Межквартильный размах: 5.1011 - 6.9975
Минимальная ошибка: 4.6488
Максимальная ошибка: 14.4342

## === СТАТИСТИЧЕСКИЕ СРАВНЕНИЯ МОДЕЛЕЙ ===
### Уровень значимости alpha = 0.05

### Парные t-тесты Стьюдента для зависимых выборок:

### Сравнение brody vs gompertz:
t-статистика = -0.4082
p-value = 0.704000
Различия не достигли уровня статистической значимости

### Сравнение brody vs logistic:
t-статистика = 0.4082
p-value = 0.704000
Различия не достигли уровня статистической значимости

### Сравнение gompertz vs logistic:
t-статистика = 0.4082
p-value = 0.704000
Различия не достигли уровня статистической значимости

## Непараметрические тесты Вилкоксона для зависимых выборок:

### Сравнение brody vs gompertz:
W-статистика = 6.0000
p-value = 0.812500
Различия не достигли уровня статистической значимости

### Сравнение brody vs logistic:
W-статистика = 6.0000
p-value = 0.812500
Различия не достигли уровня статистической значимости

### Сравнение gompertz vs logistic:
W-статистика = 6.0000
p-value = 0.812500
Различия не достигли уровня статистической значимости

Analysis completed successfully. Results saved to final_results.json

Process finished with exit code 0






