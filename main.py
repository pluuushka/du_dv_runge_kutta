from runge_kutta import *
import matplotlib.pyplot as plt

A=2/11
B=3/13
pi = np.pi

eps = 1e-4

# 1.2
h_by_first_step = first_step(0, 1e-4, 2)
result_with_first_step = integrate_system(h_by_first_step)
x_vals_first_step = [point[0] for point in result_with_first_step]
y1_vals_first_step = [point[1][0] for point in result_with_first_step]
y2_vals_first_step = [point[1][1] for point in result_with_first_step]


print("====== WITH ALGORITHM TO TAKE CHOOSE FIRST STEP ======= ")
print(f"x_k = {x_vals_first_step[-1]}")
print(f"y1(pi) = {y1_vals_first_step[-1]}")
print(f"y2(pi) = {y2_vals_first_step[-1]}")
print(f"h_by_first_step = {h_by_first_step}")

def estimate_error(h):
    traj_h = integrate_system(h)
    traj_h2 = integrate_system(h/2)
    
    error = 0
    for i, (x, y_h) in enumerate(traj_h):
        for x2, y_h2 in traj_h2:
            if abs(x - x2) < 1e-10: 
                current_error = np.linalg.norm(y_h - y_h2) # divisor is (2^s-1), where s = 2
                error = max(error, current_error)
                break
    
    return error / (2**2 - 1)

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