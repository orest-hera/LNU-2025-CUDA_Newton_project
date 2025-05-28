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
matrix_sizes = range(100, 2801, 300)

def load_and_process_data():
    # Зберігати всі дані для total_statistic
    total_stats = []
    
    # Зберігати детальні дані для GPU та cuDSS
    gpu_details = defaultdict(list)
    cudss_details = defaultdict(list)
    
    for run, dir_name in enumerate(result_dirs, 1):
        dir_path = os.path.join(base_dir, dir_name)
        if not os.path.exists(dir_path):
            continue
            
        # Завантажити total_statistic.csv (CPU=0)
        total_file = os.path.join(dir_path, "total_statistic.csv")
        if os.path.exists(total_file):
            df = pd.read_csv(total_file)
            df['run'] = run
            total_stats.append(df)
        
        # Завантажити детальні дані
        for size in matrix_sizes:
            # GPU дані (з inverse_jacobian_t)
            gpu_file = os.path.join(dir_path, f"gpu_newton_solver_{size}.csv")
            if os.path.exists(gpu_file):
                gpu_df = pd.read_csv(gpu_file)
                gpu_df['run'] = run
                gpu_details[size].append(gpu_df)
            
            # cuDSS дані (без inverse_jacobian_t)
            cudss_file = os.path.join(dir_path, f"gpu_cudss_newton_solver_{size}.csv")
            if os.path.exists(cudss_file):
                cudss_df = pd.read_csv(cudss_file)
                cudss_df['run'] = run
                cudss_details[size].append(cudss_df)
    
    # Об'єднати всі total_statistic дані
    total_df = pd.concat(total_stats) if total_stats else pd.DataFrame()
    
    # Обробка детальних даних з урахуванням різних стовпців
    def process_details(details_dict, has_inverse=True):
        processed = {}
        for size, dfs in details_dict.items():
            if dfs:
                combined = pd.concat(dfs)
                # Вибірка тільки числових стовпців
                numeric_cols = combined.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'run' and col != 'matrix_size']
                
                # Обчислити середнє та стандартне відхилення
                mean = combined.groupby('matrix_size')[numeric_cols].mean()
                std = combined.groupby('matrix_size')[numeric_cols].std()
                processed[size] = {'mean': mean, 'std': std}
        return processed
    
    gpu_processed = process_details(gpu_details, has_inverse=True)
    cudss_processed = process_details(cudss_details, has_inverse=False)
    
    return total_df, gpu_processed, cudss_processed

def plot_gpu_stats(total_df):
    if total_df.empty:
        print("No GPU/cuDSS statistics data found.")
        return
    
    # Видалити CPU (якщо всі значення 0)
    if (total_df['CPU'] == 0).all():
        total_df = total_df.drop(columns=['CPU'])
    
    # Обчислити середнє та стандартне відхилення
    stats = total_df.groupby('matrix_size').agg(['mean', 'std']).reset_index()
    
    # Побудувати графік часу виконання
    plt.figure()
    for method in ['GPU', 'cuDSS']:
        if method in stats.columns:
            plt.errorbar(
                stats['matrix_size'], 
                stats[(method, 'mean')], 
                yerr=stats[(method, 'std')],
                label=method,
                capsize=5,
                marker='o',
                linestyle='-'
            )
    
    plt.title("Час виконання для GPU та cuDSS")
    plt.xlabel("Розмір матриці")
    plt.ylabel("Час виконання (секунди)")
    plt.legend()
    plt.grid(True)
    plt.savefig("gpu_performance_comparison.png", bbox_inches='tight')
    plt.close()

def plot_phase_times(gpu_data, cudss_data):
    # Визначити етапи для кожного методу
    gpu_phases = ['func_value_t', 'jacobian_value_t', 'inverse_jacobian_t', 'delta_value_t', 'update_points_t']
    cudss_phases = ['func_value_t', 'jacobian_value_t', 'delta_value_t', 'update_points_t']
    
    # Створити графіки для GPU
    if gpu_data:
        plt.figure(figsize=(12, 6))
        
        # Пройти через всі етапи, які можуть бути в даних
        for phase in gpu_phases:
            sizes = []
            means = []
            stds = []
            
            for size in sorted(gpu_data.keys()):
                # Перевірити, чи є цей етап у даних для поточного розміру
                if phase in gpu_data[size]['mean'].columns:
                    sizes.append(size)
                    means.append(gpu_data[size]['mean'][phase].values[0])
                    stds.append(gpu_data[size]['std'][phase].values[0])
            
            if sizes:
                label = phase.replace('_t', '').replace('_', ' ')
                # Зробимо лінію для inverse_jacobian товстішою для наочності
                linewidth = 3 if phase == 'inverse_jacobian_t' else 2
                plt.errorbar(
                    sizes, means, yerr=stds,
                    label=label,
                    capsize=5,
                    marker='o',
                    linestyle='-',
                    linewidth=linewidth
                )
        
        plt.title("Час виконання окремих етапів для GPU (з inverse_jacobian)")
        plt.xlabel("Розмір матриці")
        plt.ylabel("Час виконання (секунди)")
        plt.legend()
        plt.grid(True)
        plt.savefig("gpu_phase_times.png", bbox_inches='tight', dpi=300)
        plt.close()
    
    # Створити графіки для cuDSS (без inverse_jacobian)
    if cudss_data:
        plt.figure(figsize=(12, 6))
        for phase in cudss_phases:
            sizes = []
            means = []
            stds = []
            
            for size in sorted(cudss_data.keys()):
                if phase in cudss_data[size]['mean'].columns:
                    sizes.append(size)
                    means.append(cudss_data[size]['mean'][phase].values[0])
                    stds.append(cudss_data[size]['std'][phase].values[0])
            
            if sizes:
                plt.errorbar(
                    sizes, means, yerr=stds,
                    label=phase.replace('_t', '').replace('_', ' '),
                    capsize=5,
                    marker='o',
                    linestyle='-'
                )
        
        plt.title("Час виконання окремих етапів для cuDSS")
        plt.xlabel("Розмір матриці")
        plt.ylabel("Час виконання (секунди)")
        plt.legend()
        plt.grid(True)
        plt.savefig("cudss_phase_times.png", bbox_inches='tight', dpi=300)
        plt.close()

def main():
    print("Обробка даних...")
    total_df, gpu_data, cudss_data = load_and_process_data()
    
    if not total_df.empty:
        print("Створення графіків...")
        plot_gpu_stats(total_df)
    else:
        print("Не знайдено даних total_statistic.csv")
    
    if gpu_data or cudss_data:
        plot_phase_times(gpu_data, cudss_data)
    else:
        print("Не знайдено детальних даних про час виконання етапів")
    
    print("Обробка завершена. Графіки збережено у поточній директорії.")

if __name__ == "__main__":
    main()