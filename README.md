# sheep
![image](https://github.com/user-attachments/assets/c56f425c-8da2-4ae1-805c-a4a0835ec4b3)
![image](https://github.com/user-attachments/assets/218843ad-fd5e-456d-8e30-920609a45abc)
![image](https://github.com/user-attachments/assets/50ff141e-e00f-4591-8653-6f682e144a98)
![image](https://github.com/user-attachments/assets/684d6ff6-2ec8-4740-acd2-5a3703fa5d18)
  Данные успешно сохранены в synthetic_santa_ines_m.json
  Loading data...

## === Optimizing brody model ===
Iter 10: Method=Bandit | RMSE=0.9586 | Best params=[2.59615598e+01 8.67584331e-01 1.21272276e-02]
Iter 20: Method=DEEP | RMSE=0.7883 | Best params=[2.59529239e+01 9.09724359e-01 1.10135906e-02]
Iter 30: Method=DEEP | RMSE=0.7883 | Best params=[2.59529239e+01 9.09724359e-01 1.10135906e-02]
Iter 40: Method=Bandit | RMSE=0.7244 | Best params=[2.61308127e+01 8.93640301e-01 1.08832126e-02]
Iter 50: Method=Bandit | RMSE=0.7244 | Best params=[2.61308127e+01 8.93640301e-01 1.08832126e-02]

### Results for brody:
Parameters: [2.61308127e+01 8.93640301e-01 1.08832126e-02]
RMSE: 0.7244 kg
Standard deviation: 0.7220 kg

## === Optimizing gompertz model ===
Iter 10: Method=DEEP | RMSE=0.5380 | Best params=[2.49744403e+01 2.06236499e+00 2.04981679e-02]
Iter 20: Method=DEEP | RMSE=0.5313 | Best params=[2.49744403e+01 2.12820822e+00 2.04981679e-02]
Iter 30: Method=Bandit | RMSE=0.5017 | Best params=[2.50761619e+01 2.01598629e+00 1.87767544e-02]
Iter 40: Method=Bandit | RMSE=0.4902 | Best params=[2.51261483e+01 2.00438103e+00 1.88238840e-02]
Iter 50: Method=Bandit | RMSE=0.4880 | Best params=[2.51331076e+01 2.02402094e+00 1.91536112e-02]

### Results for gompertz:
Parameters: [2.51331076e+01 2.02402094e+00 1.91536112e-02]
RMSE: 0.4880 kg
Standard deviation: 0.4880 kg

## === Optimizing logistic model ===
Iter 10: Method=DEEP | RMSE=0.9206 | Best params=[2.49158787e+01 4.50687195e+00 2.39928543e-02]
Iter 20: Method=DEEP | RMSE=0.7223 | Best params=[24.65659445  4.98063331  0.02827268]
Iter 30: Method=DEEP | RMSE=0.7223 | Best params=[24.65659445  4.98063331  0.02827268]
Iter 40: Method=Bandit | RMSE=0.7208 | Best params=[24.65659445  4.98110244  0.02926851]
Iter 50: Method=DEEP | RMSE=0.7142 | Best params=[24.52597202  4.98769864  0.029021  ]

### Results for logistic:
Parameters: [24.52597202  4.98769864  0.029021  ]
RMSE: 0.7142 kg
Standard deviation: 0.7101 kg

## === Model Comparison ===

### Расчет бутстреп-доверительных интервалов...

## === ОСНОВНЫЕ СТАТИСТИКИ МОДЕЛЕЙ ===

### Модель: brody
--------------------------------
Средняя ошибка: 0.6029
95% доверительный интервал: [0.2830, 0.9534]
Стандартное отклонение: 0.4014
Медианная ошибка: 0.6156
Межквартильный размах: 0.4019 - 0.7214
Минимальная ошибка: 0.0261
Максимальная ошибка: 1.2498

### Модель: gompertz
--------------------------------
Средняя ошибка: 0.3601
95% доверительный интервал: [0.0866, 0.6381]
Стандартное отклонение: 0.3294
Медианная ошибка: 0.1365
Межквартильный размах: 0.1269 - 0.7533
Минимальная ошибка: 0.0167
Максимальная ошибка: 0.7669

### Модель: logistic
--------------------------------
Средняя ошибка: 0.6570
95% доверительный интервал: [0.4025, 0.8841]
Стандартное отклонение: 0.2802
Медианная ошибка: 0.7520
Межквартильный размах: 0.4228 - 0.7621
Минимальная ошибка: 0.2760
Максимальная ошибка: 1.0721

## === СТАТИСТИЧЕСКИЕ СРАВНЕНИЯ МОДЕЛЕЙ ===
### Уровень значимости alpha = 0.05

Парные t-тесты Стьюдента для зависимых выборок:

### Сравнение brody vs gompertz:
t-статистика = 0.9931
p-value = 0.376877
Различия не достигли уровня статистической значимости

### Сравнение brody vs logistic:
t-статистика = -0.1875
p-value = 0.860421
Различия не достигли уровня статистической значимости

### Сравнение gompertz vs logistic:
t-статистика = -2.3444
p-value = 0.078993
Различия не достигли уровня статистической значимости

## Непараметрические тесты Вилкоксона для зависимых выборок:

### Сравнение brody vs gompertz:
W-статистика = 5.0000
p-value = 0.625000
Различия не достигли уровня статистической значимости

### Сравнение brody vs logistic:
W-статистика = 7.0000
p-value = 1.000000
Различия не достигли уровня статистической значимости

### Сравнение gompertz vs logistic:
W-статистика = 1.0000
p-value = 0.125000
Различия не достигли уровня статистической значимости

Analysis completed successfully. Results saved to final_results.json





