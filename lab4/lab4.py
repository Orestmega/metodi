import numpy as np
import matplotlib.pyplot as plt

# Функція вологості ґрунту
def M(t):
    return 50 * np.exp(-0.1 * t) + 5 * np.sin(t)

# Аналітична похідна функції
def M_prime_exact(t):
    return -5 * np.exp(-0.1 * t) + 5 * np.cos(t)

t0 = 1.0 
exact_val = M_prime_exact(t0)
print(f"1. Точне значення похідної в точці t0={t0}: {exact_val:.7f}")

# Функція для чисельного диференціювання
def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

#Дослідження залежності похибки від кроку h
hs = np.logspace(-20, 3, 1000) # Кроки від 10^-20 до 10^3
errors = np.abs(central_diff(M, t0, hs) - exact_val)

#оптимальний крок h0
opt_idx = np.argmin(errors)
h0 = hs[opt_idx]
R0 = errors[opt_idx]

print(f"2. Оптимальний крок h0: {h0:.2e}")
print(f"   Похибка при h0 (R0): {R0:.2e}")

plt.figure(figsize=(10, 6))
plt.loglog(hs, errors, label='Похибка чисельного диференціювання')
plt.axvline(h0, color='r', linestyle='--', label=f'Оптимальний крок h0={h0:.1e}')
plt.xlabel('Крок h')
plt.ylabel('Похибка R')
plt.title('Залежність похибки чисельного диференціювання від кроку h')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.gca().invert_xaxis() 
plt.show()

#Фіксується крок h = 10^-3
h = 1e-3
print(f"\n3. Вибрано фіксований крок h: {h}")

#Обчислення з кроками h та 2h
D_h = central_diff(M, t0, h)
D_2h = (M(t0 + 2*h) - M(t0 - 2*h)) / (4 * h)
print(f"4. Похідна з кроком h:  {D_h:.7f}")
print(f"   Похідна з кроком 2h: {D_2h:.7f}")

#Похибка при кроці h 
R1 = np.abs(D_h - exact_val)
print(f"5. Похибка при кроці h (R1): {R1:.7e}")

#Метод Рунге-Ромберга
y_R = D_h + (D_h - D_2h) / 3
R2 = np.abs(y_R - exact_val)
print(f"6. Уточнене значення (Рунге-Ромберг): {y_R:.7f}")
print(f"   Похибка (R2): {R2:.7e}")
print(f"   Характер зміни: похибка зменшилась у {R1/R2:.1f} разів.")

#Метод Ейткена
D_4h = (M(t0 + 4*h) - M(t0 - 4*h)) / (8 * h)

# Уточнене значення похідної за Ейткеном
numerator = (D_2h)**2 - D_4h * D_h
denominator = 2 * D_2h - (D_4h + D_h)
y_E = numerator / denominator

# Порядок точності p
p = (1 / np.log(2)) * np.log(np.abs((D_4h - D_2h) / (D_2h - D_h)))

R3 = np.abs(y_E - exact_val)

print(f"7. Уточнене значення (Ейткен): {y_E:.7f}")
print(f"   Порядок точності (p): {p:.2f}")
print(f"   Похибка (R3): {R3:.7e}")
print(f"   Характер зміни: похибка Ейткена зменшилась порівняно з R1 у {R1/R3:.1f} разів.")