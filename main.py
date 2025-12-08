from runge_kutta import *
import matplotlib.pyplot as plt
from task3 import main_task3

A=2/11
B=3/13
pi = np.pi

eps = 1e-4
s = 2

# 1.2
h_by_first_step = first_step(0, eps, s)
result_with_first_step = integrate_system(h_by_first_step)
x_vals_first_step = [point[0] for point in result_with_first_step]
y1_vals_first_step = [point[1][0] for point in result_with_first_step]
y2_vals_first_step = [point[1][1] for point in result_with_first_step]


print("====== WITH ALGORITHM TO TAKE CHOOSE FIRST STEP ======= ")
print(f"x_k = {x_vals_first_step[-1]}")
print(f"y1(pi) = {y1_vals_first_step[-1]}")
print(f"y2(pi) = {y2_vals_first_step[-1]}")
print(f"h_by_first_step = {h_by_first_step}")


print(f"full error = {estimate_error(h_by_first_step)}")

plt.figure(figsize=(10, 5))
plt.plot(x_vals_first_step, y1_vals_first_step, label='y₁(x)')
plt.plot(x_vals_first_step, y2_vals_first_step, label='y₂(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solve the system with first step')
plt.grid(True)
plt.show()

# 2.1 
x0 = 0
y0 = np.array([B * pi, A * pi])

x_final, y_final = auto_step(x0, y0, eps, s)

# Compare with fixed step for verification
print("\n" + "=" * 70)
print("COMPARISON WITH FIXED STEP SIZE")
print("=" * 70)

h_fixed = first_step(x0, eps, s)
result_fixed = integrate_system(h_fixed)
x_fixed = result_fixed[-1][0]
y_fixed = result_fixed[-1][1]

print(f"Fixed step size h = {h_fixed:.6f}")
print(f"Result with fixed step:")
print(f"  x_final = {x_fixed:.6f}, y_final = [{y_fixed[0]}, {y_fixed[1]:.15e}]")
print(f"\nResult with automatic step:")
print(f"  x_final = {x_final:.6f}, y_final = [{y_final[0]}, {y_final[1]:.15e}]")

# Usage function first_step from runge_kutta.py 
# with eps = 1e-4 and s = 4 for 3 volume

h_opt_2nd = first_step(0, 1e-4, 2)
h_opt_4th = first_step(0, 1e-4, 4)

errors_2nd = integrate_with_errors_2nd(h_opt_2nd)
errors_4th = integrate_with_errors_4th(h_opt_4th)

x_vals_2nd = [point[0] for point in errors_2nd]
error_vals_2nd = [point[1] for point in errors_2nd]

x_vals_4th = [point[0] for point in errors_4th]
error_vals_4th = [point[1] for point in errors_4th]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))


ax1 = axes[0, 0]
ax1.semilogy(x_vals_2nd, error_vals_2nd, 'b-', linewidth=1.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('Истинная полная погрешность', fontsize=12)
ax1.set_title(f'Метод 2-го порядка\nh = {h_opt_2nd:.4e}, ε = {eps}', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=eps, color='r', linestyle='--', alpha=0.7, label=f'Целевая точность ε = {eps}')
ax1.legend(fontsize=10)


ax2 = axes[0, 1]
ax2.semilogy(x_vals_4th, error_vals_4th, 'r-', linewidth=1.5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('Истинная полная погрешность', fontsize=12)
ax2.set_title(f'Метод 4-го порядка\nh = {h_opt_4th:.4e}, ε = {eps}', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=eps, color='r', linestyle='--', alpha=0.7, label=f'Целевая точность ε = {eps}')
ax2.legend(fontsize=10)


ax3 = axes[1, 0]
ax3.semilogy(x_vals_2nd, error_vals_2nd, 'b-', label='Метод 2-го порядка', linewidth=1.5)
ax3.semilogy(x_vals_4th, error_vals_4th, 'r-', label='Метод 4-го порядка', linewidth=1.5)
ax3.set_xlabel('x', fontsize=12)
ax3.set_ylabel('Истинная полная погрешность', fontsize=12)
ax3.set_title('Сравнение погрешностей методов', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)


ax4 = axes[1, 1]
methods = ['Метод 2-го порядка', 'Метод 4-го порядка']
h_values = [h_opt_2nd, h_opt_4th]
colors = ['blue', 'red']

bars = ax4.bar(methods, h_values, color=colors, alpha=0.7)
ax4.set_ylabel('Оптимальный шаг h', fontsize=12)
ax4.set_title('Сравнение оптимальных шагов', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')


for bar, h in zip(bars, h_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{h:.2e}', ha='center', va='bottom', fontsize=11)

plt.suptitle('ЗАДАНИЕ 3.2: Зависимость истинной полной погрешности от x', fontsize=16)
plt.tight_layout()
plt.show()

main_task3()