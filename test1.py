import numpy as np
import matplotlib.pyplot as plt
import requests
import math

raw_coords = [
    (48.164214, 24.536044), (48.164983, 24.534836), (48.165605, 24.534068),
    (48.166228, 24.532915), (48.166777, 24.531927), (48.167326, 24.530884),
    (48.167011, 24.530061), (48.166053, 24.528039), (48.166655, 24.526064),
    (48.166497, 24.523574), (48.166128, 24.520214), (48.165416, 24.517170),
    (48.164546, 24.514640), (48.163412, 24.512980), (48.162331, 24.511715),
    (48.162015, 24.509462), (48.162147, 24.506932), (48.161751, 24.504244),
    (48.161197, 24.501793), (48.160580, 24.500537), (48.160250, 24.500106)
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def get_elevations(coords):
    locations = "|".join([f"{lat},{lon}" for lat, lon in coords])
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={locations}"
    try:
        print("Запит до API висот...")
        resp = requests.get(url, timeout=10)
        data = resp.json()['results']
        return [point['elevation'] for point in data]
    except Exception as e:
        print(f"API помилка ({e}). Використовую дані-заглушки.")
        return [1200 + i * (800/len(coords)) + np.random.normal(0, 5) for i in range(len(coords))]

class CubicSpline:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.h = np.diff(self.x)
        self.a = self.y[:]
        self.b = np.zeros(self.n)
        self.c = np.zeros(self.n)
        self.d = np.zeros(self.n)
        self.solve_tridiagonal_system()
        self.calc_remaining_coeffs()

    def solve_tridiagonal_system(self):
        n = self.n
        alpha = np.zeros(n)
        beta = np.zeros(n)
        gamma = np.zeros(n)
        delta = np.zeros(n)
        A = np.zeros(n)
        B = np.zeros(n)
        self.c[0] = 0.0
        self.c[n-1] = 0.0
        
        for i in range(1, n-1):
            alpha[i] = self.h[i-1]
            beta[i]  = 2 * (self.h[i-1] + self.h[i])
            gamma[i] = self.h[i]
            term1 = (self.y[i+1] - self.y[i]) / self.h[i]
            term2 = (self.y[i] - self.y[i-1]) / self.h[i-1]
            delta[i] = 3 * (term1 - term2)
            
        A[1] = -gamma[1] / beta[1]
        B[1] = delta[1] / beta[1]
        for i in range(2, n-1):
            denom = beta[i] + alpha[i] * A[i-1]
            A[i] = -gamma[i] / denom
            B[i] = (delta[i] - alpha[i] * B[i-1]) / denom
        for i in range(n-2, 0, -1):
            self.c[i] = A[i] * self.c[i+1] + B[i]

    def calc_remaining_coeffs(self):
        for i in range(self.n - 1):
            self.d[i] = (self.c[i+1] - self.c[i]) / (3 * self.h[i])
            self.b[i] = ((self.y[i+1] - self.y[i]) / self.h[i]) - (self.h[i] / 3) * (self.c[i+1] + 2 * self.c[i])

    def interpolate(self, x_val):
        if x_val < self.x[0] or x_val > self.x[-1]: return 0
        i = 0
        for idx in range(len(self.x) - 1):
            if self.x[idx] <= x_val <= self.x[idx+1]:
                i = idx
                break
        dx = x_val - self.x[i]
        return self.a[i] + self.b[i]*dx + self.c[i]*(dx**2) + self.d[i]*(dx**3)


def main():
    elevations = get_elevations(raw_coords)
    
    distances = [0.0]
    cum_dist = 0.0
    for i in range(1, len(raw_coords)):
        d = haversine(raw_coords[i-1][0], raw_coords[i-1][1], raw_coords[i][0], raw_coords[i][1])
        cum_dist += d
        distances.append(cum_dist)
    
    distances = np.array(distances)
    elevations = np.array(elevations)
    max_nodes = len(raw_coords)

    # хз
    print(f"\nВсього доступно точок: {max_nodes}")
    try:
        user_input = input(f"Введіть кількість вузлів для наближення (від 2 до {max_nodes}): ")
        n_approx = int(user_input)
        if n_approx < 2 or n_approx > max_nodes:
            n_approx = 5
            print("Некоректне число, взято за замовчуванням 5.")
    except ValueError:
        n_approx = 5
        print("Помилка вводу, взято за замовчуванням 5.")

    full_spline = CubicSpline(distances, elevations)
    
    indices = np.linspace(0, max_nodes-1, n_approx, dtype=int)
    x_approx_nodes = distances[indices]
    y_approx_nodes = elevations[indices]
    approx_spline = CubicSpline(x_approx_nodes, y_approx_nodes)

    x_smooth = np.linspace(distances[0], distances[-1], 500)
    y_true_vals = np.array([full_spline.interpolate(x) for x in x_smooth]) #
    y_approx_vals = np.array([approx_spline.interpolate(x) for x in x_smooth]) #
    error_vals = np.abs(y_true_vals - y_approx_vals) #меджик

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    ax1.plot(distances, elevations, 'ro', label='Вихідні вузли (GPS)', markersize=5) 
    ax1.plot(x_smooth, y_true_vals, 'b-', label='Кубічний сплайн (всі точки)')
    ax1.plot(x_smooth, y_approx_vals, 'g--', label=f'Сплайн ({n_approx} вузлів)', linewidth=2)
    
    ax1.set_ylabel("Висота (м)")
    ax1.set_title("Профіль висоти маршруту: Заросляк - Говерла")
    ax1.legend()
    ax1.grid(True)

    # графік Похибки
    ax2.plot(x_smooth, error_vals, 'r-', label='Похибка')
    ax2.fill_between(x_smooth, error_vals, color='red', alpha=0.1)
    
    ax2.set_xlabel("Відстань (м)")
    ax2.set_ylabel("Абсолютна похибка (м)")
    ax2.set_title(f"Графік похибки (Макс: {np.max(error_vals):.2f} м)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    print(f"\n--- Результати для {n_approx} вузлів ---")
    print(f"Максимальна похибка: {np.max(error_vals):.4f} м")

if __name__ == "__main__":
    main()