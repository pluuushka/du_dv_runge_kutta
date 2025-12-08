import numpy as np
import matplotlib.pyplot as plt
from runge_kutta import *

def task_3_3_1_2():

    print("=" * 80)
    print("АНАЛИЗ АВТОМАТИЧЕСКОГО ВЫБОРА ШАГА")
    print("=" * 80)
    
    eps = 1e-4
    x0 = 0
    y0 = np.array([B * pi, A * pi])
    
    # Запускаем оба метода с автоматическим выбором шага
    print("\n" + "=" * 40)
    print("МЕТОД 2-ГО ПОРЯДКА")
    print("=" * 40)
    
    # Используем новую функцию
    x_final_2nd, y_final_2nd, steps_2nd, ratios_2nd, iterations_2nd = \
        get_auto_step_data(x0, y0, eps, method='2nd')
    
    print(f"Результат: x_final = {x_final_2nd:.6f}, y_final = {y_final_2nd}")
    print(f"Количество итераций: {iterations_2nd}")
    
    print("\n" + "=" * 40)
    print("МЕТОД 4-ГО ПОРЯДКА")
    print("=" * 40)
    
    x_final_4th, y_final_4th, steps_4th, ratios_4th, iterations_4th = \
        get_auto_step_data(x0, y0, eps, method='4th')
    
    print(f"Результат: x_final = {x_final_4th:.6f}, y_final = {y_final_4th}")
    print(f"Количество итераций: {iterations_4th}")
    
    # Извлекаем данные для графиков
    step_x_2nd = [point[0] for point in steps_2nd]
    step_h_2nd = [point[1] for point in steps_2nd]
    
    step_x_4th = [point[0] for point in steps_4th]
    step_h_4th = [point[1] for point in steps_4th]
    
    ratio_x_2nd = [point[0] for point in ratios_2nd]
    ratio_2nd = [point[1] for point in ratios_2nd]
    
    ratio_x_4th = [point[0] for point in ratios_4th]
    ratio_4th = [point[1] for point in ratios_4th]
    
    # Построение графиков 3.3.1 и 3.3.2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # График 3.3.1: Шаг интегрирования для метода 2-го порядка
    ax1 = axes[0, 0]
    ax1.semilogy(step_x_2nd, step_h_2nd, 'b.-', markersize=4, linewidth=1)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Шаг интегрирования h', fontsize=12)
    ax1.set_title('3.3.1: Метод 2-го порядка\nЗависимость шага от x', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # График 3.3.1: Шаг интегрирования для метода 4-го порядка
    ax2 = axes[0, 1]
    ax2.semilogy(step_x_4th, step_h_4th, 'r.-', markersize=4, linewidth=1)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Шаг интегрирования h', fontsize=12)
    ax2.set_title('3.3.1: Метод 4-го порядка\nЗависимость шага от x', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # График 3.3.2: Отношение погрешностей для метода 2-го порядка
    ax3 = axes[1, 0]
    ax3.plot(ratio_x_2nd, ratio_2nd, 'b.-', markersize=4, linewidth=1)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Идеальное отношение = 1')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Отношение (истинная/оценка)', fontsize=12)
    ax3.set_title('3.3.2: Метод 2-го порядка\nОтношение погрешностей', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Устанавливаем разумные пределы для оси Y
    if ratio_2nd:
        y_min = min(0, min(ratio_2nd) * 0.9)
        y_max = max(2, max(ratio_2nd) * 1.1)
        ax3.set_ylim([y_min, y_max])
    
    # График 3.3.2: Отношение погрешностей для метода 4-го порядка
    ax4 = axes[1, 1]
    ax4.plot(ratio_x_4th, ratio_4th, 'r.-', markersize=4, linewidth=1)
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='Идеальное отношение = 1')
    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('Отношение (истинная/оценка)', fontsize=12)
    ax4.set_title('3.3.2: Метод 4-го порядка\nОтношение погрешностей', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Устанавливаем разумные пределы для оси Y
    if ratio_4th:
        y_min = min(0, min(ratio_4th) * 0.9)
        y_max = max(2, max(ratio_4th) * 1.1)
        ax4.set_ylim([y_min, y_max])
    
    plt.suptitle('Анализ автоматического выбора шага (ε=10⁻⁴)', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Статистика
    print("\n" + "=" * 80)
    print("СТАТИСТИКА ДЛЯ ε = 10⁻⁴:")
    print("=" * 80)
    print(f"{'Параметр':<35} {'Метод 2-го порядка':<25} {'Метод 4-го порядка':<25}")
    print("-" * 85)
    print(f"{'Количество шагов':<35} {len(steps_2nd):<25} {len(steps_4th):<25}")
    print(f"{'Минимальный шаг':<35} {min(step_h_2nd) if step_h_2nd else 0:<25.2e} {min(step_h_4th) if step_h_4th else 0:<25.2e}")
    print(f"{'Максимальный шаг':<35} {max(step_h_2nd) if step_h_2nd else 0:<25.2e} {max(step_h_4th) if step_h_4th else 0:<25.2e}")
    print(f"{'Средний шаг':<35} {np.mean(step_h_2nd) if step_h_2nd else 0:<25.2e} {np.mean(step_h_4th) if step_h_4th else 0:<25.2e}")
    print(f"{'Среднее отношение погр.':<35} {np.mean(ratio_2nd) if ratio_2nd else 0:<25.3f} {np.mean(ratio_4th) if ratio_4th else 0:<25.3f}")
    print(f"{'Количество итераций':<35} {iterations_2nd:<25} {iterations_4th:<25}")
    
    return steps_2nd, steps_4th, ratios_2nd, ratios_4th

def analyze_adaptive_step_behavior():
    """
    Дополнительный анализ поведения автоматического выбора шага
    """
    print("\n" + "=" * 80)
    print("ПОВЕДЕНИЕ АВТОМАТИЧЕСКОГО ВЫБОРА ШАГА")
    print("=" * 80)
    
    eps = 1e-4
    x0 = 0
    y0 = np.array([B * pi, A * pi])
    
    # Анализируем распределение шагов
    print("\nРаспределение шагов для метода 4-го порядка (ε = 10⁻⁴):")
    
    _, _, steps_4th, _, _ = auto_step_4th(x0, y0, eps, s=4, ro=eps)
    step_h_4th = [point[1] for point in steps_4th]
    
    # Статистика по шагам
    if step_h_4th:
        print(f"  Всего различных шагов: {len(set([round(h, 6) for h in step_h_4th]))}")
        print(f"  Минимальный шаг: {min(step_h_4th):.2e}")
        print(f"  Максимальный шаг: {max(step_h_4th):.2e}")
        print(f"  Медианный шаг: {np.median(step_h_4th):.2e}")
        print(f"  Средний шаг: {np.mean(step_h_4th):.2e}")
        print(f"  Стандартное отклонение: {np.std(step_h_4th):.2e}")
        
        # Гистограмма распределения шагов
        plt.figure(figsize=(10, 6))
        plt.hist(step_h_4th, bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Значение шага h', fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.title('Распределение шагов интегрирования (метод 4-го порядка)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Анализируем, как часто срабатывает каждый случай
    print("\nАнализ частоты случаев в алгоритме выбора шага:")
    
    def analyze_cases_frequency(method='4th'):
        
        if method == '2nd':
            s = 2
            step_func = runge_kutta
        else:
            s = 4
            step_func = runge_kutta_4th
        
        h = first_step(x0, eps, s)
        x = x0
        y = y0
        
        case_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        while x < pi:
            y1_step = step_func(x, y, h)
            y_half_step1 = step_func(x, y, h/2)
            y_half_step2 = step_func(x + h/2, y_half_step1, h/2)
            
            rho = norm(y1_step - y_half_step2) / (2**s - 1)
            
            if rho > eps * 2**s:
                case_counts[1] += 1
                h /= 2
                continue
            elif eps < rho <= eps * 2**s:
                case_counts[2] += 1
                y = y_half_step2
                x = x + h
                h_next = h / 2
            elif eps / 2**(s+1) <= rho <= eps:
                case_counts[3] += 1
                y = y1_step
                x = x + h
                h_next = h
            else:
                case_counts[4] += 1
                y = y1_step
                x = x + h
                h_next = 2 * h
            
            h = h_next
            
            if x + h > pi:
                h = pi - x
            
            if x >= pi:
                break
        
        total = sum(case_counts.values())
        if total > 0:
            for case, count in case_counts.items():
                percentage = count / total * 100
                print(f"  Случай {case}: {count} раз ({percentage:.1f}%)")
        
        return case_counts
    
    print("\nДля метода 2-го порядка:")
    cases_2nd = analyze_cases_frequency('2nd')
    
    print("\nДля метода 4-го порядка:")
    cases_4th = analyze_cases_frequency('4th')

def task_3_3_3():
    print("\n" + "=" * 80)
    print("ЗАВИСИМОСТЬ КОЛИЧЕСТВА ВЫЧИСЛЕНИЙ ОТ ТОЧНОСТИ")
    print("=" * 80)
    
    # Используем функцию из runge_kutta.py
    print("\nАнализ метода 2-го порядка...")
    epsilons_2nd, f_calls_2nd = analyze_epsilon_dependence(method='2nd')
    
    print("\nАнализ метода 4-го порядка...")
    epsilons_4th, f_calls_4th = analyze_epsilon_dependence(method='4th')
    
    # Выводим таблицу результатов
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 80)
    print(f"{'ε':<10} {'Вычисления (2-й)':<20} {'Вычисления (4-й)':<20} {'Отношение (2-й/4-й)':<20}")
    print("-" * 80)
    
    for i in range(len(epsilons_2nd)):
        ratio = f_calls_2nd[i] / f_calls_4th[i] if f_calls_4th[i] > 0 else 0
        print(f"{epsilons_2nd[i]:<10.1e} {f_calls_2nd[i]:<20} {f_calls_4th[i]:<20} {ratio:<20.2f}")
    
    # Построение графиков 3.3.3
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График 1: Зависимость вычислений от точности
    ax1 = axes[0]
    ax1.loglog(epsilons_2nd, f_calls_2nd, 'bo-', linewidth=2, markersize=8, label='Метод 2-го порядка')
    ax1.loglog(epsilons_4th, f_calls_4th, 'ro-', linewidth=2, markersize=8, label='Метод 4-го порядка')
    
    ax1.set_xlabel('Точность ε', fontsize=12)
    ax1.set_ylabel('Оценочное количество вычислений f', fontsize=12)
    ax1.set_title('3.3.3: Зависимость вычислений от точности', fontsize=14)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=11)
    
    # График 2: Отношение вычислений
    ax2 = axes[1]
    ratio_values = [f_calls_2nd[i] / f_calls_4th[i] for i in range(len(epsilons_2nd))]
    
    ax2.plot(epsilons_2nd, ratio_values, 'g-o', linewidth=2, markersize=8)
    ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Равная эффективность')
    ax2.set_xlabel('Точность ε', fontsize=12)
    ax2.set_ylabel('Отношение вычислений (2-й/4-й)', fontsize=12)
    ax2.set_title('Отношение вычислительных затрат', fontsize=14)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.suptitle('Зависимость вычислений правой части от точности', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return epsilons_2nd, f_calls_2nd, epsilons_4th, f_calls_4th

def main_task3():


    steps_2nd, steps_4th, ratios_2nd, ratios_4th = task_3_3_1_2()

    epsilons_2nd, f_calls_2nd, epsilons_4th, f_calls_4th = task_3_3_3()
    
    analyze_adaptive_step_behavior()

if __name__ == "__main__":
    main_task3()