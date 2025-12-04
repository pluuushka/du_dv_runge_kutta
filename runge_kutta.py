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

# 1.2 
def first_step(x0, eps, s):
    norm_f0 = np.linalg.norm(f(x0, [B * np.pi, A * np.pi])) # x0 = 0, xk = pi
    delta = (1 / np.pi) ** (s + 1) + (norm_f0)**(s + 1) # s = 2

    h1 = (eps / delta)**(1/(s+1))

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
            x = pi
        else:
            y = runge_kutta(x, y, h)
            x += h
        trajectory.append((x, y.copy()))
    return trajectory


# 2.1 

def auto_step(x0, y0, eps, s, ro=1e-5):

    h = first_step(x0, eps, s)
    x = x0
    y = y0
    
    iteration = 0
    print("=" * 70)
    print("INTEGRATION WITH AUTOMATIC STEP SIZE SELECTION")
    print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
    print(f"Parameters: ε = {eps}, ro = {ro}, order s = {s}")
    print(f"Initial step: h = {h}")
    print("=" * 70)
    
    while x < pi:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")
        print(f"Current point: x = {x:.6f}, y = {y}")
        print(f"Current step: h = {h:.6f}")
        
        # Compute approximations
        y1_step = runge_kutta(x, y, h)
        y_half_step1 = runge_kutta(x, y, h/2)
        y_half_step2 = runge_kutta(x + h/2, y_half_step1, h/2)
        
        rho = norm(y1_step - y_half_step2) / (2**s - 1)
        print(f"Local error estimate ρ = {rho:.2e}")
        print(f"Thresholds: ε·2^s = {ro * 2**s:.2e}, ε = {ro:.2e}, ε/2^(s+1) = {ro / 2**(s+1):.2e}")
        
        # Determine which case applies
        if rho > ro * 2**s: # the situation where is local error is to big, we're need to take more small step, like h/2
            print("CASE 1: ρ > ε·2^s")
            print("  Reason: Error is too large")
            print("  Action: Halve the step and recalculate at the same point")
            h /= 2
            print(f"  New step: h = {h:.6f}")
            print("  Approximation rejected, recalculating with new step")
            continue
        
        elif ro < rho <= ro * 2**s: # also bad, but we can take half-step solution
            print("CASE 2: ε < ρ ≤ ε·2^s")
            print("  Reason: Error exceeds tolerance but is within limits")
            print("  Action: Accept the two-step approximation (y_half_step2)")
            y = y_half_step2
            x_old = x
            x = x + h
            h_next = h / 2
            print(f"  Transition: x: {x_old:.6f} → {x:.6f}")
            print(f"  Accepted y = {y}")
            print(f"  Next step: h = {h_next:.6f}")
        
        elif ro / 2**(s+1) <= rho <= ro: # vse na mazi, kaifuem
            print("CASE 3: ε/2^(s+1) ≤ ρ ≤ ε")
            print("  Reason: Error is within acceptable limits")
            print("  Action: Accept one-step approximation, keep step unchanged")
            y = y1_step
            x_old = x
            x = x + h
            h_next = h
            print(f"  Transition: x: {x_old:.6f} → {x:.6f}")
            print(f"  Accepted y = {y}")
            print(f"  Step unchanged: h = {h_next:.6f}")
        
        else:  # rho < ro / 2**(s+1) ocheno malo... nado brat' bol'she
            print("CASE 4: ρ < ε/2^(s+1)")
            print("  Reason: Error is significantly below tolerance")
            print("  Action: Accept one-step approximation, double the step")
            y = y1_step
            x_old = x
            x = x + h
            h_next = 2 * h
            print(f"  Transition: x: {x_old:.6f} → {x:.6f}")
            print(f"  Accepted y = {y}")
            print(f"  Step doubled: h = {h_next:.6f}")
        
        # Update step for next iteration
        h = h_next
        
        # Check if we would exceed the boundary
        if x + h > pi:
            h = pi - x
            print(f"\nWARNING: Approaching boundary π, reducing step to {h:.6f}")
        
        print(f"  Distance to π: {pi - x:.6f}")
        
        if x >= pi:
            print(f"\nReached final point x = {x:.6f}")
            break
    
    print("\n" + "=" * 70)
    print("INTEGRATION COMPLETED")
    print(f"Total iterations: {iteration}")
    print(f"Final point: x = {x:.6f}, y = {y}")
    print("=" * 70)
    
    return x, y


def runge_kutta_4th(x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h/2, y + k1/2)
    k3 = h * f(x + h/2, y + k2/2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

# analyctical solution
def exact_solution(x):
    omega = np.sqrt(A * B)
    y1 = B * pi * np.cos(omega * x) + (A**2 * pi / omega) * np.sin(omega * x)
    y2 = -(B * pi * omega / A) * np.sin(omega * x) + A * pi * np.cos(omega * x)
    return np.array([y1, y2])

def auto_step_4th(x0, y0, eps, s=4, ro=1e-5, track_data=True):
    h = first_step(x0, eps, s)
    x = x0
    y = y0
    
    iteration = 0
    step_history = []
    error_ratio_history = []
    
    if track_data:
        print("=" * 70)
        print("AUTO STEP FOR 4TH ORDER METHOD")
        print(f"Initial conditions: x0 = {x0}, y0 = {y0}")
        print(f"Parameters: ε = {eps}, ro = {ro}, order s = {s}")
        print(f"Initial step: h = {h}")
        print("=" * 70)
    
    while x < pi:
        iteration += 1
        
        # Compute approximations
        y1_step = runge_kutta_4th(x, y, h)
        y_half_step1 = runge_kutta_4th(x, y, h/2)
        y_half_step2 = runge_kutta_4th(x + h/2, y_half_step1, h/2)
        
        # Estimate local error
        rho = norm(y1_step - y_half_step2) / (2**s - 1)
        
        # True local error
        y_exact = exact_solution(x + h)
        true_error = norm(y1_step - y_exact)
        
        # Store data if tracking
        if track_data:
            step_history.append((x, h))
            if rho > 0:
                error_ratio_history.append((x, true_error / rho))
            else:
                error_ratio_history.append((x, 0))
        
        # Determine which case applies
        if rho > ro * 2**s:
            h /= 2
            continue
        elif ro < rho <= ro * 2**s:
            y = y_half_step2
            x = x + h
            h_next = h / 2
        elif ro / 2**(s+1) <= rho <= ro:
            y = y1_step
            x = x + h
            h_next = h
        else:  # rho < ro / 2**(s+1)
            y = y1_step
            x = x + h
            h_next = 2 * h
        
        h = h_next
        
        # Check if we would exceed the boundary
        if x + h > pi:
            h = pi - x
        
        if x >= pi:
            break
    
    if track_data:
        print("\n" + "=" * 70)
        print("INTEGRATION COMPLETED")
        print(f"Total iterations: {iteration}")
        print(f"Final point: x = {x:.6f}, y = {y}")
        print("=" * 70)
    
    if track_data:
        return x, y, step_history, error_ratio_history, iteration
    else:
        return x, y


def integrate_system_4th(h=0.0001):
    x = 0
    y = np.array([B * pi, A * pi])  # start condition
    trajectory = [(x, y.copy())]
    while x < pi:
        if ((x + h) > pi):
            step_error = pi - x
            y = runge_kutta_4th(x, y, step_error)
            x = pi
        else:
            y = runge_kutta_4th(x, y, h)
            x += h
        trajectory.append((x, y.copy()))
    return trajectory



def integrate_with_errors_4th(h):
    x = 0
    
    y = np.array([B * pi, A * pi])
    
    errors = [] # (x, error)
    
    while x < pi:
        if x + h > pi:
            h_actual = pi - x
        else:
            h_actual = h
        
      
        y_new = runge_kutta_4th(x, y, h_actual)
        
        x_new = x + h_actual
        
        y_exact = exact_solution(x_new)
        
        full_error = norm(y_new - y_exact)
        errors.append((x_new, full_error))
        
        x = x_new
        y = y_new
    
    return errors

def integrate_with_errors_2nd(h):
    x = 0
    
    y = np.array([B * pi, A * pi])
    
    errors = [] # (x, error)
    
    while x < pi:
        if x + h > pi:
            h_actual = pi - x
        else:
            h_actual = h
        
      
        y_new = runge_kutta(x, y, h_actual)
        
        x_new = x + h_actual
        
        y_exact = exact_solution(x_new)
        
        full_error = norm(y_new - y_exact)
        errors.append((x_new, full_error))
        
        x = x_new
        y = y_new
    
    return errors

def estimate_error_4th(h):
    traj_h = integrate_system_4th(h)
    traj_h2 = integrate_system_4th(h/2)
    
    error = 0
    for (x, y_h) in enumerate(traj_h):
        for x2, y_h2 in traj_h2:
            if abs(x - x2) < 1e-10: 
                current_error = np.linalg.norm(y_h - y_h2) # divisor is (2^s-1), where s = 2
                error = max(error, current_error)
                break
    
    return error / (2**4 - 1)





