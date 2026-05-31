import numpy as np
import time
import multiprocessing as mp
import matplotlib.pyplot as plt

# епанечниково ядро
def epanechnikov_kernel(u):
    u = np.asarray(u)
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)

# вспомогательные функции для параллельных процессов
def calc_numerator(weight_array, target_values, result_container):
    result_container.value = float(np.sum(weight_array * target_values))

def calc_denominator(weight_array, result_container):
    result_container.value = float(np.sum(weight_array))

# последовательная реализация (эталон)
def sequential_implementation(weight_array, target_values):
    start_time = time.perf_counter()
    numerator = 0.0
    for idx in range(len(weight_array)):
        numerator += weight_array[idx] * target_values[idx]
    denominator = 0.0
    for idx in range(len(weight_array)):
        denominator += weight_array[idx]
    return numerator, denominator, time.perf_counter() - start_time

# параллельная версия с 2 процессами (числитель и знаменатель параллельно)
def parallel_two_processes(weight_array, target_values):
    num_container = mp.Value('d', 0.0)
    den_container = mp.Value('d', 0.0)
    proc_num = mp.Process(target=calc_numerator, 
                          args=(weight_array, target_values, num_container))
    proc_den = mp.Process(target=calc_denominator, 
                          args=(weight_array, den_container))
    start_time = time.perf_counter()
    proc_num.start()
    proc_den.start()
    proc_num.join()
    proc_den.join()
    return num_container.value, den_container.value, time.perf_counter() - start_time

# параллельная версия с 4 процессами (числитель и знаменатель по двум половинам массива)
def parallel_four_processes(weight_array, target_values):
    midpoint = len(weight_array) // 2
    num_left = mp.Value('d', 0.0)
    num_right = mp.Value('d', 0.0)
    den_left = mp.Value('d', 0.0)
    den_right = mp.Value('d', 0.0)
    proc_num_left = mp.Process(target=calc_numerator, 
                               args=(weight_array[:midpoint], target_values[:midpoint], num_left))
    proc_num_right = mp.Process(target=calc_numerator, 
                                args=(weight_array[midpoint:], target_values[midpoint:], num_right))
    proc_den_left = mp.Process(target=calc_denominator, 
                               args=(weight_array[:midpoint], den_left))
    proc_den_right = mp.Process(target=calc_denominator, 
                                args=(weight_array[midpoint:], den_right))
    start_time = time.perf_counter()
    for proc in (proc_num_left, proc_num_right, proc_den_left, proc_den_right):
        proc.start()
    for proc in (proc_num_left, proc_num_right, proc_den_left, proc_den_right):
        proc.join()
    total_num = num_left.value + num_right.value
    total_den = den_left.value + den_right.value
    return total_num, total_den, time.perf_counter() - start_time

# измерение среднего времени для всех точек сетки
def benchmark_times(x_data, y_data, beta_param, delta_param, x_grid_points):
    times_seq = []
    times_2proc = []
    times_4proc = []
    bandwidth = delta_param / beta_param
    for x0 in x_grid_points:
        weights = epanechnikov_kernel((x0 - x_data) / bandwidth)
        if np.sum(weights) == 0:
            continue
        _, _, t_seq = sequential_implementation(weights, y_data)
        _, _, t_2p = parallel_two_processes(weights, y_data)
        _, _, t_4p = parallel_four_processes(weights, y_data)
        times_seq.append(t_seq)
        times_2proc.append(t_2p)
        times_4proc.append(t_4p)
    return np.mean(times_seq), np.mean(times_2proc), np.mean(times_4proc)

# главная функция
def main():
    print("доступно CPU ядер:", mp.cpu_count())
    # генерация данных
    np.random.seed(69)
    sample_size = 5_000_000
    x_coords = np.random.uniform(-3, 7, sample_size)
    noise_std = 0.1 + 0.2 * np.abs(np.sin(x_coords))
    y_values = (0.1 * (x_coords - 4) * np.cos(x_coords) + 
                0.5 * np.sin(1.5 * x_coords) +
                0.2 * np.cos(0.5 * x_coords**2) + 
                0.05 * x_coords**2 +
                np.random.normal(0, noise_std, sample_size))
    sorted_x = np.sort(x_coords)
    delta_value = np.max(np.diff(sorted_x))
    beta_value = 1.0
    grid_points = np.linspace(-3, 7, 20)
    # замер производительности
    t_seq, t_2p, t_4p = benchmark_times(x_coords, y_values, beta_value, delta_value, grid_points)
    speedup_2p = t_seq / t_2p
    speedup_4p = t_seq / t_4p
    # вывод результатов
    print("результаты измерений:")
    print(f"  последовательный:    {t_seq:.6f} сек")
    print(f"  2 процесса:          {t_2p:.6f} сек  (ускорение = {speedup_2p:.3f})")
    print(f"  4 процесса:          {t_4p:.6f} сек  (ускорение = {speedup_4p:.3f})")
    # график времени выполнения
    plt.figure(figsize=(7, 5))
    plt.bar(["1 процесс", "2 процесса", "4 процесса"], [t_seq, t_2p, t_4p], 
            color=["blue", "green", "orange"])
    plt.ylabel("время (сек)")
    plt.title("сравнение времени выполнения")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()
    # график ускорения
    plt.figure(figsize=(7, 5))
    plt.bar(["2 процесса", "4 процесса"], [speedup_2p, speedup_4p], 
            color=["green", "orange"])
    plt.axhline(y=1, color="gray", linestyle="--")
    plt.ylabel("ускорение")
    plt.title("ускорение относительно 1 процесса")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
