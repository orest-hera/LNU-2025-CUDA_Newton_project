import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

N_measured = np.array([1, 2, 4, 5, 8, 12, 20])
mem_7nz = np.array([85, 98, 146, 176, 324, 620, 1540])
mem_14nz = np.array([89, 106, 171, 220, 429, 835, 2168])

p7 = Polynomial.fit(N_measured, mem_7nz, deg=2)
p14 = Polynomial.fit(N_measured, mem_14nz, deg=2)

N_predict = np.linspace(1, 70, 500)
mem7_predict = p7(N_predict)
mem14_predict = p14(N_predict)

plt.figure(figsize=(12, 7))

plt.scatter(N_measured, mem_7nz, color='blue', label='7 nz (measured)', zorder=5)
plt.scatter(N_measured, mem_14nz, color='orange', label='14 nz (measured)', zorder=5)

plt.plot(N_predict, mem7_predict, 'b--', label='7 nz (quadratic fit, forecast)')
plt.plot(N_predict, mem14_predict, 'orange', linestyle='--', label='14 nz (quadratic fit, forecast)')

plt.title('Forecast of GPU Memory Usage up to 100k Rows')
plt.xlabel('Number of Rows (in thousands)')
plt.ylabel('Memory Increase (MB)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

c7 = p7.convert().coef
c14 = p14.convert().coef
print(f"7 nz approx: M ≈ {c7[2]:.3f}·N² + {c7[1]:.3f}·N + {c7[0]:.3f}")
print(f"14 nz approx: M ≈ {c14[2]:.3f}·N² + {c14[1]:.3f}·N + {c14[0]:.3f}")
