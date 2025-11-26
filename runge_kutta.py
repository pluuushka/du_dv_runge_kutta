import numpy as np

c2 = 1/9
A = 2/11
B= 3/13

def generate_a21_b1_b2():
    a21 = c2
    b2 = 1 / (2*c2)
    b1 = 1 - b2
    return a21, b2, b1

def f(x, y): 
    return np.array([A * y[1], -B * y[0]])

def runge_kutta(x,y, h):
    a21, b2, b1 = generate_a21_b1_b2()
    k1 = h * f(x, y)
    k2 = h * f(x + c2 * h, y + a21 * k1)
    
    return y + b1 * k1 + b2 * k2


# 1.2 
def first_step(x0, eps, s):
    norm_f0 = np.linalg.norm(f(x0, [B * np.pi, A * np.pi])) # x0 = 0, xk = pi
    delta = (1 / np.pi) ** (s + 1) + (norm_f0)**(s + 1) # s = 2

    h1 = (eps / delta)**(1/s+1)

    if norm_f0 < 1e-10:
        y_eiler = [B * np.pi, A * np.pi] + h1 * f(x0, [B * np.pi, A * np.pi])
        x_eiler = x0 + h1

        f_euler = f(x_eiler, y_eiler)
        norm_f_euler = np.linalg.norm(f_euler)
        delta_euler = (1.0 / max(abs(x_eiler), abs(np.pi))) ** (s + 1) + norm_f_euler ** (s + 1)
        h2 = (eps/ delta_euler) ** (1.0 / (s + 1))


        h = min(h1, h2)
    else:
        h = h1
    
    return h

