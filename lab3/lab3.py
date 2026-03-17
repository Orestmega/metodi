import csv
import math
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'temperature_data.csv'
sample_data = [
    [1,-2], [2,0], [3,5], [4,10], [5,15], [6,20], [7,23], [8,22], 
    [9,17], [10,10], [11,5], [12,0], [13,-10], [14,3], [15,7], [16,13], 
    [17,19], [18,20], [19,22], [20,21], [21,18], [22,15], [23,10], [24,3]
]

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Month", "Temp"])
    writer.writerows(sample_data)

x_data = []
y_data = []
with open(csv_filename, mode='r') as file:
    reader = csv.reader(file)
    next(reader) 
    for row in reader:
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

n = len(x_data) - 1 

#Функції МНК

def form_matrix(x, m):
    """Формування матриці A розміром (m+1, m+1)"""
    A = np.zeros((m+1, m+1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = sum((xi ** (i + j)) for xi in x)
    return A

def form_vector(x, y, m):
    """Формування вектора вільних членів b розміром (m+1)"""
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = sum(yi * (xi ** i) for xi, yi in zip(x, y))
    return b

def gauss_solve(A_in, b_in):
    """Метод Гауса з вибором головного елемента по стовпцю"""
    A = np.copy(A_in)
    b = np.copy(b_in)
    size = len(b)
    
    # Прямий хід
    for k in range(size - 1):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        if A[max_row, k] == 0:
            continue

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, size):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
            
    # Зворотній хід
    x_sol = np.zeros(size)
    for i in range(size - 1, -1, -1):
        sum_ax = sum(A[i, j] * x_sol[j] for j in range(i + 1, size))
        x_sol[i] = (b[i] - sum_ax) / A[i, i]
        
    return x_sol

def polynomial(x_vals, coef):
    """Обчислення значень многочлена"""
    y_poly = np.zeros(len(x_vals))
    for i in range(len(coef)):
        y_poly += coef[i] * (np.array(x_vals) ** i)
    return y_poly

def variance(y_true, y_approx):
    """Обчислення дисперсії (похибки)"""
    n_points = len(y_true)
    sum_sq = sum((yt - ya) ** 2 for yt, ya in zip(y_true, y_approx))
    return math.sqrt(sum_sq / n_points)

#залежність дисперсії від степення многочлена
variances = []
best_m = 1
min_var = float('inf')
best_coefs = []

print("=== Аналіз похибок для різних степенів многочлена ===")
# Перевіряємо m від 1 до 10
for m in range(1, 11):
    A = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    try:
        coef = gauss_solve(A, b_vec)
        y_approx = polynomial(x_data, coef)
        var = variance(y_data, y_approx)
        variances.append(var)
        print(f"Степінь m={m}: Дисперсія = {var:.4f}")
        
        if var < min_var:
            min_var = var
            best_m = m
            best_coefs = coef
    except Exception as e:
         print(f"Степінь m={m}: Помилка обчислення (матриця вироджена або близька до неї)")
         variances.append(float('inf'))

print(f"\n=> Оптимальний степінь многочлена: m = {best_m} (Дисперсія: {min_var:.4f})")

# гарфік похибки
y_approx_best = polynomial(x_data, best_coefs)
errors = np.abs(np.array(y_data) - y_approx_best)

#Прогноз на наступні 3 місяці
x_future = [25, 26, 27]
y_future = polynomial(x_future, best_coefs)

print("\n=== Прогноз на наступні 3 місяці ===")
for month, temp in zip(x_future, y_future):
    print(f"Місяць {month}: {temp:.2f} градусів")

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(range(1, 11), variances, marker='o', color='purple')
plt.title('Залежність дисперсії від степеня апроксимуючого многочлена')
plt.xlabel('Степінь m')
plt.ylabel('Дисперсія')
plt.grid(True)
plt.xticks(range(1, 11))

plt.subplot(3, 1, 2)
x_smooth = np.linspace(min(x_data), max(x_future), 200)
y_smooth = polynomial(x_smooth, best_coefs)

plt.scatter(x_data, y_data, color='blue', label='Фактичні дані (з CSV)')
plt.plot(x_smooth, y_smooth, color='red', label=f'Апроксимація (m={best_m})')
plt.scatter(x_future, y_future, color='green', marker='*', s=150, label='Прогноз (екстраполяція)')
plt.title('Апроксимація температурних даних методом найменших квадратів')
plt.xlabel('Місяць')
plt.ylabel('Температура')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.bar(x_data, errors, color='orange')
plt.title('Абсолютна похибка апроксимації у вузлах')
plt.xlabel('Місяць')
plt.ylabel('Похибка ε(x)')
plt.grid(axis='y')
plt.xticks(x_data)

plt.tight_layout()
plt.show()