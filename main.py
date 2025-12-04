from runge_kutta import *
import matplotlib.pyplot as plt

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
