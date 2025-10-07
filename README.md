
<img width="1580" height="222" alt="image" src="https://github.com/user-attachments/assets/a5d52df7-18c6-4c83-9769-6f2b8e98f87b" />
# Demonstrasi Penyelesaian Sistem Persamaan Non-Linear (Materi M06)

Repositori ini berisi implementasi beberapa metode numerik dalam bahasa Python untuk menyelesaikan sistem persamaan non-linear sebagai tugas mata kuliah Metode Numerik.

## 📝 Deskripsi Masalah

Sistem persamaan non-linear yang akan diselesaikan adalah sebagai berikut:

\$f_1(x, y) = x^2 + xy - 10 = 0\$

\$f_2(x, y) = y + 3xy^2 - 57 = 0\$

Dengan kondisi:
* **Tebakan Awal**: \$x_0 = 1.5\$, \$y_0 = 3.5\$
* **Toleransi Galat (epsilon)**: \$0.000001\$

## ⚙️ Metode yang Digunakan

Sesuai dengan ketentuan tugas untuk **NIMx = 0**, implementasi ini mencakup empat metode berikut:

1.  **Iterasi Titik Tetap (Metode Jacobi)**: Menggunakan kombinasi fungsi iterasi \`g1A\` dan \`g2A\`.
2.  **Iterasi Titik Tetap (Metode Gauss-Seidel)**: Menggunakan kombinasi fungsi iterasi \`g1A\` dan \`g2A\`.
3.  **Metode Newton-Raphson**: Metode berbasis turunan untuk menemukan akar dengan konvergensi kuadratik.
4.  **Metode Secant**: Aproksimasi dari metode Newton-Raphson.
