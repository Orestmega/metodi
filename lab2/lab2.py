import csv                                                                                                                  # M - сума
import math                                                                                                                 # П - добуток
import matplotlib.pyplot as plt
import numpy as np

def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Tasks']))
            y.append(float(row['Cost']))
    return x, y

# нютон
def divided_differences(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y                                                                                                                                     # Записуємо в перший стовпець (індекс 0) нашої матриці всі значення ігреків.
    for j in range(1, n):
        for i in range(n - j):                                                                                                                       # Цикл по рядках (з кожним стовпцем кількість рядків зменшується на 1).                            
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])                                                                   # Рахуємо саму різницю: (нижнє значення - верхнє значення) / (різниця крайніх іксів)
    return coef[0, :]

# поліноми
def newton_polynomial(coef, x_data, x): 
    n = len(x_data) - 1                                                                                                                             # визначає степінь полінома
    p = coef[n]
    for k in range(1, n + 1):
        p = coef[n - k] + (x - x_data[n - k]) * p
    return p

#WW
def omega_function(x, x_data):
    result = 1.0
    for xi in x_data:
        result *= (x - xi)
    return result

def finite_differences(y):
    n = len(y)
    delta = np.zeros([n, n])
    delta[:, 0] = y                                                                                                                                 # Перший стовпець - це y
    for j in range(1, n):
        for i in range(n - j):
            delta[i][j] = delta[i + 1][j - 1] - delta[i][j - 1]                                                                                     # віднімаються сусідні значення
    return delta[0, :]

def factorial_polynomial(delta_y0, x_data, x):
    h = x_data[1] - x_data[0]
    t = (x - x_data[0]) / h
    n = len(delta_y0)
    
    result = delta_y0[0]
    t_k = 1.0
    for k in range(1, n):
        t_k *= (t - k + 1) 
        result += (delta_y0[k] * t_k) / math.factorial(k)
    return result

print("ЧАСТИНА 1:")
try:
    x_data, y_data = read_data("data.csv")
    print(f"Вузли (Tasks): {x_data}")
    print(f"Значення (Cost): {y_data}\n")

    target_x = 15000
    coef_5 = divided_differences(x_data, y_data)
    cost_15000 = newton_polynomial(coef_5, x_data, target_x)
    print(f"Прогноз вартості для {target_x} завдань (5 вузлів): ${cost_15000:.4f}")

    x_3, y_3 = x_data[:3], y_data[:3]
    x_4, y_4 = x_data[:4], y_data[:4]

    coef_3 = divided_differences(x_3, y_3)
    coef_4 = divided_differences(x_4, y_4)

    print(f"Прогноз (3 вузли): ${newton_polynomial(coef_3, x_3, target_x):.4f}")
    print(f"Прогноз (4 вузли): ${newton_polynomial(coef_4, x_4, target_x):.4f}\n")

    x_smooth = np.linspace(min(x_data), max(x_data), 500)
    y_smooth_3 = [newton_polynomial(coef_3, x_3, xi) for xi in x_smooth]
    y_smooth_4 = [newton_polynomial(coef_4, x_4, xi) for xi in x_smooth]
    y_smooth_5 = [newton_polynomial(coef_5, x_data, xi) for xi in x_smooth]

    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=8, label='Експериментальні дані')
    plt.plot(target_x, cost_15000, 'r*', markersize=12, label=f'Прогноз ({target_x} tasks)')
    plt.plot(x_smooth, y_smooth_3, 'g--', label='3 вузли')
    plt.plot(x_smooth, y_smooth_4, 'b-.', label='4 вузли')
    plt.plot(x_smooth, y_smooth_5, 'm-', label='5 вузлів (повна модель)')
    
    plt.title('Прогноз вартості обчислень (Cost = f(Tasks))')
    plt.xlabel('Кількість завдань (Tasks)')
    plt.ylabel('Вартість, $ (Cost)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Табулювання за методичкою ---")
    a = min(x_data)
    b = max(x_data)
    n_nodes = len(x_data)
    h = (b - a) / (20 * n_nodes)
    print(f"Відрізок: [{a}, {b}]")
    print(f"Крок табуляції h = {h}")
    
    x_tab = np.arange(a, b + h, h)
    N_val, w_val, eps_val = [], [], []
    
    for x in x_tab:
        nx = newton_polynomial(coef_5, x_data, x)
        N_val.append(nx)
        w_val.append(omega_function(x, x_data))
        
        # Рахуємо похибку e(x) як різницю між прогнозом по 5 і 4 точках
        nx_4 = newton_polynomial(coef_4, x_4, x)
        eps_val.append(abs(nx - nx_4))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    ax1.plot(x_tab, N_val, 'b-', label='N_n(x)')
    ax1.plot(x_data, y_data, 'ro', label='Вузли')
    ax1.set_title('Інтерполяційний многочлен Ньютона')
    ax1.grid(True); ax1.legend()

    ax2.plot(x_tab, eps_val, 'r-', label='Похибка ε(x)')
    ax2.set_title('Похибка ε(x) = |N_5(x) - N_4(x)|')
    ax2.grid(True); ax2.legend()

    ax3.plot(x_tab, w_val, 'g-', label='w_n(x)')
    ax3.axhline(0, color='black', linewidth=1) 
    ax3.plot(x_data, [0]*len(x_data), 'ro') 
    ax3.set_title('Функція вузлів ω_n(x)')
    ax3.grid(True); ax3.legend()

    plt.tight_layout()
    plt.show()
    # -----------------------------------------------------

except FileNotFoundError:
    print("Файл data.csv не знайдено.")

print("\nЧАСТИНА 2:")

x_values_granular = np.linspace(0, 1, 128)
y_values_real = np.sin(x_values_granular * np.pi * 5)

node_counts = [5, 10, 20]

plt.figure(figsize=(12, 8))
plt.plot(x_values_granular, y_values_real, 'k-', linewidth=2, label='Реальна функція (sin)')
#Рунге
for n in node_counts:
    x_nodes = np.linspace(0, 1, n)
    y_nodes = np.sin(x_nodes * np.pi * 5)
    
    print(f"--- Кількість вузлів: {n} ---")
    print(f"Координати іксів вузлів:\n{np.round(x_nodes, 4)}")
    
    coef_newton = divided_differences(x_nodes, y_nodes)
    y_interp_newton = [newton_polynomial(coef_newton, x_nodes, xi) for xi in x_values_granular]

    delta_y0 = finite_differences(y_nodes)
    y_interp_fact = [factorial_polynomial(delta_y0, x_nodes, xi) for xi in x_values_granular]
    
    max_error_newton = np.max(np.abs(y_values_real - y_interp_newton))
    max_error_fact = np.max(np.abs(y_values_real - y_interp_fact))
    
    print(f"Максимальна похибка (Ньютон): {max_error_newton:.6f}")
    print(f"Максимальна похибка (Факторіальний): {max_error_fact:.6f}\n")
 
    plt.plot(x_values_granular, y_interp_newton, '--', label=f'Інтерполяція (n={n})')
    plt.plot(x_nodes, y_nodes, 'o', markersize=6)

plt.title('Аналіз ефекту Рунге: порівняння для 5, 10 та 20 вузлів')
plt.xlabel('x')
plt.ylabel('y')
plt.ylim(-2.5, 2.5) 
plt.legend()
plt.grid(True)
plt.show()