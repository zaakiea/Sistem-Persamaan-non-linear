import numpy as np

# --- Definisi Fungsi Dasar ---
def f1(x, y):
    """Fungsi pertama f1(x, y) = x^2 + x*y - 10"""
    return x**2 + x*y - 10

def f2(x, y):
    """Fungsi kedua f2(x, y) = y + 3*x*y^2 - 57"""
    return y + 3*x*y**2 - 57

# --- Fungsi Iterasi Titik Tetap untuk NIMx = 0 ---
def g1A(x, y):
    """g1A: x = sqrt(10 - x*y)"""
    arg = 10 - x*y
    if arg < 0:
        return np.nan
    return np.sqrt(arg)

def g2A(x, y):
    """g2A: y = sqrt(57 - 3*x*y^2)"""
    arg = 57 - 3*x*(y**2)
    if arg < 0:
        return np.nan
    return np.sqrt(arg)

# --- Metode Iterasi Titik Tetap: Jacobi ---
def fixed_point_jacobi(x0, y0, tol, max_iter=200):
    x, y = x0, y0
    print("\n--- Menjalankan Iterasi Titik Tetap (Jacobi) dengan g1A & g2A ---")
    for i in range(max_iter):
        x_new = g1A(x, y)
        y_new = g2A(x, y)
        if np.isnan(x_new) or np.isnan(y_new):
            print(f"Iterasi {i+1}: domain keluar (nilai negatif di sqrt). Iterasi dihentikan.")
            return None, None, i+1
        error = max(abs(x_new - x), abs(y_new - y))
        print(f"Iterasi {i+1}: x = {x_new:.9f}, y = {y_new:.9f}, error = {error:.3e}")
        if error < tol:
            print(f"Solusi (Jacobi) ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
        x, y = x_new, y_new
    print("Jacobi: tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

# --- Metode Iterasi Titik Tetap: Gauss-Seidel ---
def fixed_point_seidel(x0, y0, tol, max_iter=200):
    x, y = x0, y0
    print("\n--- Menjalankan Iterasi Titik Tetap (Gauss-Seidel) dengan g1A & g2A ---")
    for i in range(max_iter):
        x_old, y_old = x, y
        x = g1A(x_old, y_old)
        if np.isnan(x):
            print(f"Iterasi {i+1}: domain keluar saat menghitung x (sqrt negatif). Hentikan.")
            return None, None, i+1
        # gunakan x baru untuk menghitung y (Seidel)
        y = g2A(x, y_old)
        if np.isnan(y):
            print(f"Iterasi {i+1}: domain keluar saat menghitung y (sqrt negatif). Hentikan.")
            return None, None, i+1
        error = max(abs(x - x_old), abs(y - y_old))
        print(f"Iterasi {i+1}: x = {x:.9f}, y = {y:.9f}, error = {error:.3e}")
        if error < tol:
            print(f"Solusi (Seidel) ditemukan setelah {i+1} iterasi.")
            return x, y, i+1
    print("Seidel: tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

# --- Metode Newton-Raphson ---
def df1dx(x, y): return 2*x + y
def df1dy(x, y): return x
def df2dx(x, y): return 3*(y**2)
def df2dy(x, y): return 1 + 6*x*y

def newton_raphson(x0, y0, tol, max_iter=100):
    x, y = x0, y0
    print("\n--- Menjalankan Metode Newton-Raphson ---")
    for i in range(max_iter):
        J = np.array([[df1dx(x, y), df1dy(x, y)],
                      [df2dx(x, y), df2dy(x, y)]], dtype=float)
        F = np.array([-f1(x, y), -f2(x, y)], dtype=float)
        try:
            delta = np.linalg.solve(J, F)
        except np.linalg.LinAlgError:
            print(f"Iterasi {i+1}: Jacobian singular. Metode gagal.")
            return None, None, i+1
        x_new, y_new = x + delta[0], y + delta[1]
        err = np.sqrt(delta[0]**2 + delta[1]**2)
        print(f"Iterasi {i+1}: x = {x_new:.12f}, y = {y_new:.12f}, error = {err:.3e}")
        if err < tol:
            print(f"Solusi (Newton) ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
        x, y = x_new, y_new
    print("Newton: tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

# --- Metode Secant (aproks. Jacobian dengan beda hingga) ---
def secant_quasi(x0, y0, tol, max_iter=200, h=1e-6):
    x, y = x0, y0
    print("\n--- Menjalankan Metode Secant (aproks. Jacobian beda hingga) ---")
    for i in range(max_iter):
        # aproksimasi turunan (forward difference)
        j11 = (f1(x + h, y) - f1(x, y)) / h
        j12 = (f1(x, y + h) - f1(x, y)) / h
        j21 = (f2(x + h, y) - f2(x, y)) / h
        j22 = (f2(x, y + h) - f2(x, y)) / h
        J_approx = np.array([[j11, j12], [j21, j22]], dtype=float)
        F = np.array([-f1(x, y), -f2(x, y)], dtype=float)
        try:
            delta = np.linalg.solve(J_approx, F)
        except np.linalg.LinAlgError:
            print(f"Iterasi {i+1}: aproks. Jacobian singular. Metode gagal.")
            return None, None, i+1
        x_new, y_new = x + delta[0], y + delta[1]
        err = np.sqrt(delta[0]**2 + delta[1]**2)
        print(f"Iterasi {i+1}: x = {x_new:.12f}, y = {y_new:.12f}, error = {err:.3e}")
        if err < tol:
            print(f"Solusi (Secant) ditemukan setelah {i+1} iterasi.")
            return x_new, y_new, i+1
        x, y = x_new, y_new
    print("Secant: tidak konvergen dalam batas iterasi maksimum.")
    return None, None, max_iter

# --- Eksekusi Utama ---
if __name__ == "__main__":
    # Tebakan awal (ubah bila perlu untuk eksperimen konvergensi)
    x_init, y_init = 1.5, 3.5
    tolerance = 1e-6

    # Jalankan metode
    x_jacobi, y_jacobi, iter_jacobi = fixed_point_jacobi(x_init, y_init, tolerance)
    x_seidel, y_seidel, iter_seidel = fixed_point_seidel(x_init, y_init, tolerance)
    x_newton, y_newton, iter_newton = newton_raphson(x_init, y_init, tolerance)
    x_secant, y_secant, iter_secant = secant_quasi(x_init, y_init, tolerance)

    # Rangkuman hasil
    print("\n" + "="*60)
    print("                      RANGKUMAN HASIL AKHIR")
    print("="*60)
    print(f"{'Metode':<30} {'Solusi (x, y)':<30} {'Iterasi'}")
    print("-"*90)
    def fmt(res):
        if res[0] is None:
            return f"{'Gagal/Non-konvergen':<30}"
        else:
            return f"({res[0]:.6f}, {res[1]:.6f})"
    print(f"{'IT Jacobi (g1A, g2A)':<30} {fmt((x_jacobi,y_jacobi,iter_jacobi)):<30} {iter_jacobi}")
    print(f"{'IT Seidel (g1A, g2A)':<30} {fmt((x_seidel,y_seidel,iter_seidel)):<30} {iter_seidel}")
    print(f"{'Newton-Raphson':<30} {fmt((x_newton,y_newton,iter_newton)):<30} {iter_newton}")
    print(f"{'Secant (aprox)':<30} {fmt((x_secant,y_secant,iter_secant)):<30} {iter_secant}")
    print("="*60)
