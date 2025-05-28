import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Налаштування стилю графіків
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Шлях до папок з результатами
base_dir = "../"
result_dirs = [f"results_{i}" for i in range(1, 6)]
matrix_sizes = range(100, 501, 50)

def load_and_process_data():
    # Зберігати всі дані для total_statistic
    total_stats = []
    
    # Зберігати детальні дані для кожного методу
    cpu_details = defaultdict(list)
    gpu_details = defaultdict(list)
    cudss_details = defaultdict(list)
    
    for run, dir_name in enumerate(result_dirs, 1):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
            
        # Завантажити total_statistic.csv
        total_file = os.path.join(dir_path, "total_statistic.csv")
        if os.path.exists(total_file):
            df = pd.read_csv(total_file)
            df['run'] = run
            total_stats.append(df)
        
        # Завантажити детальні дані для кожного методу
        for size in matrix_sizes:
            # CPU дані
            cpu_file = os.path.join(dir_path, f"cpu_newton_solver_{size}.csv")
            if os.path.exists(cpu_file):
                cpu_df = pd.read_csv(cpu_file)
                cpu_df['run'] = run
                cpu_details[size].append(cpu_df)
            
            # GPU дані
            gpu_file = os.path.join(dir_path, f"gpu_newton_solver_{size}.csv")
            if os.path.exists(gpu_file):
                gpu_df = pd.read_csv(gpu_file)
                gpu_df['run'] = run
                gpu_details[size].append(gpu_df)
            
            # cuDSS дані
            cudss_file = os.path.join(dir_path, f"gpu_cudss_newton_solver_{size}.csv")
            if os.path.exists(cudss_file):
                cudss_df = pd.read_csv(cudss_file)
                cudss_df['run'] = run
                cudss_details[size].append(cudss_df)
    
    # Об'єднати всі total_statistic дані
    total_df = pd.concat(total_stats) if total_stats else pd.DataFrame()
    
    # Обробка детальних даних
    def process_details(details_dict):
        processed = {}
        for size, dfs in details_dict.items():
            if dfs:
                combined = pd.concat(dfs)
                # Обчислити середнє та стандартне відхилення для кожного стовпця
                mean = combined.groupby('matrix_size').mean(numeric_only=True)
                std = combined.groupby('matrix_size').std(numeric_only=True)
                processed[size] = {'mean': mean, 'std': std}
        return processed
    
    cpu_processed = process_details(cpu_details)
    gpu_processed = process_details(gpu_details)
    cudss_processed = process_details(cudss_details)
    
    return total_df, cpu_processed, gpu_processed, cudss_processed

def plot_total_stats(total_df):
    if total_df.empty:
        print("No total statistics data found.")
        return
    
    # Обчислити середнє та стандартне відхилення для кожного розміру матриці
    stats = total_df.groupby('matrix_size').agg(['mean', 'std']).reset_index()
    
    # Побудувати графік для загального часу
    plt.figure(figsize=(12, 6))
    for method in ['CPU', 'GPU', 'cuDSS']:
        plt.errorbar(
            stats['matrix_size'], 
            stats[(method, 'mean')], 
            yerr=stats[(method, 'std')],
            label=method,
            capsize=5,
            marker='o',
            linestyle='-'
        )
    
    plt.title("Середній час виконання для різних методів з відхиленнями")
    plt.xlabel("Розмір матриці")
    plt.ylabel("Час виконання (секунди)")
    plt.legend()
    plt.grid(True)
    plt.savefig("total_performance_comparison.png")
    plt.close()

def plot_phase_times(cpu_data, gpu_data, cudss_data):
    # Створити графіки для кожного етапу обчислень
    phases = {
        'cpu': ['func_value_t', 'jacobian_value_t', 'inverse_jacobian_t', 'delta_value_t', 'update_points_t'],
        'gpu': ['func_value_t', 'jacobian_value_t', 'delta_value_t', 'update_points_t'],
        'cudss': ['func_value_t', 'jacobian_value_t', 'delta_value_t', 'update_points_t']
    }
    
    for method, data, method_name in zip(
        ['cpu', 'gpu', 'cudss'],
        [cpu_data, gpu_data, cudss_data],
        ['CPU', 'GPU', 'cuDSS']
    ):
        if not data:
            continue
            
        plt.figure(figsize=(12, 8))
        
        for phase in phases[method]:
            # Зібрати дані для цієї фази
            sizes = []
            means = []
            stds = []
            
            for size in sorted(data.keys()):
                if phase in data[size]['mean']:
                    sizes.append(size)
                    means.append(data[size]['mean'][phase].values[0])
                    stds.append(data[size]['std'][phase].values[0])
            
            if sizes:
                plt.errorbar(
                    sizes, means, yerr=stds,
                    label=phase.replace('_t', '').replace('_', ' '),
                    capsize=5,
                    marker='o',
                    linestyle='-'
                )
        
        plt.title(f"Час виконання окремих етапів для {method_name}")
        plt.xlabel("Розмір матриці")
        plt.ylabel("Час виконання (секунди)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{method_name.lower()}_phase_times.png")
        plt.close()

def plot_speedup(total_df):
    if total_df.empty:
        return
    
    stats = total_df.groupby('matrix_size').agg(['mean', 'std']).reset_index()
    
    # Обчислити прискорення GPU та cuDSS відносно CPU
    stats[('GPU_speedup', 'mean')] = stats[('CPU', 'mean')] / stats[('GPU', 'mean')]
    stats[('GPU_speedup', 'std')] = stats[('GPU_speedup', 'mean')] * np.sqrt(
        (stats[('CPU', 'std')]/stats[('CPU', 'mean')])**2 + 
        (stats[('GPU', 'std')]/stats[('GPU', 'mean')])**2
    )
    
    stats[('cuDSS_speedup', 'mean')] = stats[('CPU', 'mean')] / stats[('cuDSS', 'mean')]
    stats[('cuDSS_speedup', 'std')] = stats[('cuDSS_speedup', 'mean')] * np.sqrt(
        (stats[('CPU', 'std')]/stats[('CPU', 'mean')])**2 + 
        (stats[('cuDSS', 'std')]/stats[('cuDSS', 'mean')])**2
    )
    
    # Побудувати графік прискорення
    plt.figure(figsize=(12, 6))
    for method in ['GPU', 'cuDSS']:
        plt.errorbar(
            stats['matrix_size'],
            stats[(f'{method}_speedup', 'mean')],
            yerr=stats[(f'{method}_speedup', 'std')],
            label=f'{method} vs CPU',
            capsize=5,
            marker='o',
            linestyle='-'
        )
    
    plt.title("Прискорення GPU та cuDSS відносно CPU")
    plt.xlabel("Розмір матриці")
    plt.ylabel("Коефіцієнт прискорення")
    plt.legend()
    plt.grid(True)
    plt.savefig("speedup_comparison.png")
    plt.close()

def main():
    print("Обробка даних...")
    total_df, cpu_data, gpu_data, cudss_data = load_and_process_data()
    
    if not total_df.empty:
        print("Створення графіків...")
        plot_total_stats(total_df)
        plot_speedup(total_df)
    else:
        print("Не знайдено даних total_statistic.csv")
    
    if cpu_data or gpu_data or cudss_data:
        plot_phase_times(cpu_data, gpu_data, cudss_data)
    else:
        print("Не знайдено детальних даних про час виконання етапів")
    
    print("Обробка завершена. Графіки збережено у поточній директорії.")

if __name__ == "__main__":
    main()