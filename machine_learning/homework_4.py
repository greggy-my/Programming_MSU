import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy_loss(y, p):
    return - (y * np.log(p) + (1 - y) * np.log(1 - p))


def compute_gradients(y, p, x):
    dw0 = -((y / p) - ((1 - y) / (1 - p))) * p * (1 - p)
    dw1 = -((y / p) - ((1 - y) / (1 - p))) * x[1] * p * (1 - p)
    dw2 = -((y / p) - ((1 - y) / (1 - p))) * x[2] * p * (1 - p)
    return dw0, dw1, dw2


# Заданные значения
omega_0 = 0
omega_1 = 2
omega_2 = -1
x1 = 1
x2 = 2

# Вычисление p
z = omega_0 + omega_1 * x1 + omega_2 * x2
p = sigmoid(z)

# Вычисление производных
dw0, dw1, dw2 = compute_gradients(1, p, [1, x1, x2])

# Вывод результатов
print("Значение произведения производных:", dw0 * dw1 * dw2)
