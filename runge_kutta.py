import numpy as np
from numpy.linalg import norm

c2 = 1/9
A = 2/11
B= 3/13
pi = np.pi

def generate_a21_b1_b2():
    a21 = c2
    b2 = 1 / (2*c2)
    b1 = 1 - b2
    return a21, b2, b1

def f(x, y): 
    return np.array([A * y[1], -B * y[0]])

def runge_kutta(x, y, h):
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

def integrate_system(h=0.0001):
    x = 0
    y = np.array([B * pi, A * pi])  # start condition
    trajectory = [(x, y.copy())]
    while x < pi:
        if ((x + h) > pi):
            step_error = pi - x
            y = runge_kutta(x, y, step_error)
        x += h
        trajectory.append((x, y.copy()))
    return trajectory


# 2.1 

def auto_step(x0, y0, eps, s, ro=1e-5):

    h = first_step(x0, eps, s)
    x = x0 + h
    y = y0

    points = [(x, y.copy())]

    while x < pi:

        y = [B * np.pi, A * np.pi] + h * f(x, [B * np.pi, A * np.pi])

        y1_step = runge_kutta(x, y, h)

        y_half_step1 = runge_kutta(x, y, h/2)
        y_half_step2 = runge_kutta(x + h/2, y_half_step1, h/2)

        if norm(y1_step - y_half_step2) > ro * 2**s: # the situation when local error is too big, need to calc with new step
            h /= 2
            continue

        if ro < norm(y1_step - y_half_step2) <= ro * 2**s: # the situation when local error is too big also, but in this situation
            y = y_half_step2                                # unlike previous we can take value which was calculate by half-step
            x = x + h
            h_next = h / 2

        if ro < norm(y1_step - y_half_step2) < ro / 2**(s + 1): # the situation when local error is too small and recommended to increase step to 2h
            y = y1_step
            x = x + h
            h_next = 2 * h
        
        
        if ro / 2**(s + 1) <= norm(y1_step - y_half_step2) <= ro: # the situation when local error is satisfactory and we don't have to change the step
            y = y1_step
            x = x + h
            h_next = h

        h = h_next

    return points








