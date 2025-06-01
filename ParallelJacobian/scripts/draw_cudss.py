import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from pathlib import Path

# Налаштування стилю графіків
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_all_runs(base_dir="../"):
    """Зчитує дані з усіх папок results_*"""
    all_data = []
    phase_data = []
    
    # Знаходимо всі папки з результатами
    results_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('results_') and os.path.isdir(os.path.join(base_dir, d))])
    
    if not results_dirs:
        raise FileNotFoundError("Не знайдено папок з результатами (results_*)")
    
    for run_dir in results_dirs:
        run_path = os.path.join(base_dir, run_dir)
        stats_file = os.path.join(run_path, 'total_statistic.csv')
        
        if os.path.exists(stats_file):
            # Читаємо основні статистики
            df = pd.read_csv(stats_file)
            df['run'] = run_dir  # Додаємо ідентифікатор запуску
            all_data.append(df)
            
            # Читаємо детальні дані по фазах
            for file in os.listdir(run_path):
                if file.startswith('gpu_cudss_newton_solver_') and file.endswith('.csv'):
                    file_path = os.path.join(run_path, file)
                    phase_df = pd.read_csv(file_path)
                    phase_df['run'] = run_dir
                    
                    # Витягуємо zeros_per_row з назви файлу або даних
                    if 'zeros_per_row' not in phase_df.columns:
                        # Спробуємо витягти з назви файлу
                        try:
                            zeros = int(file.split('_')[-1].split('.')[0])
                            phase_df['zeros_per_row'] = zeros
                            phase_df['nonzeros_per_row'] = 10000 - zeros  # Додаємо кількість ненульових елементів
                        except:
                            continue
                    
                    phase_data.append(phase_df)
    
    if not all_data:
        raise ValueError("Не знайдено даних у папках results_*")
    
    # Об'єднуємо всі запуски
    combined_stats = pd.concat(all_data, ignore_index=True)
    combined_phases = pd.concat(phase_data, ignore_index=True) if phase_data else None
    
    return combined_stats, combined_phases

def analyze_cudss_performance(df):
    """Аналізує продуктивність cuDSS з урахуванням усіх запусків"""
    # Обчислюємо середні значення та стандартні відхилення
    stats = df.groupby('zeros_per_row').agg({
        'cuDSS': ['mean', 'std', 'min', 'max'],
        'run': 'count'
    }).reset_index()
    
    stats.columns = ['zeros_per_row', 'cuDSS_mean', 'cuDSS_std', 'cuDSS_min', 'cuDSS_max', 'run_count']
    
    # Додаткові метрики
    stats['nonzeros_per_row'] = 10000 - stats['zeros_per_row']
    stats['sparsity_percent'] = stats['zeros_per_row'] / 10000 * 100
    stats['density_percent'] = 100 - stats['sparsity_percent']
    
    # Сортуємо за зростанням ненульових елементів (зменшенням нулів)
    stats = stats.sort_values('nonzeros_per_row').reset_index(drop=True)
    
    return stats

def plot_performance(stats):
    """Створює графіки продуктивності"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Графік 1: Час від кількості ненульових елементів з помилками
    ax1.errorbar(stats['nonzeros_per_row'], stats['cuDSS_mean'], 
                yerr=stats['cuDSS_std'], fmt='o-', color='orange',
                capsize=5, linewidth=2, label='Середній час ± STD')
    ax1.plot(stats['nonzeros_per_row'], stats['cuDSS_min'], 'g--', label='Мінімальний час')
    ax1.plot(stats['nonzeros_per_row'], stats['cuDSS_max'], 'r--', label='Максимальний час')
    
    ax1.set_title('Продуктивність cuDSS залежно від кількості ненульових елементів\n')
    ax1.set_xlabel('Кількість ненульових елементів у рядку')
    ax1.set_ylabel('Час виконання (секунди)')
    ax1.legend()
    ax1.grid(True)
    
    # Графік 2: Відносна продуктивність (сортуємо за зростанням ненульових елементів)
    min_time = stats['cuDSS_mean'].min()
    stats['speedup'] = stats['cuDSS_mean'] / min_time
    
    # Створюємо мітки для осі X у відсотках щільності
    x_labels = [f"{int(density)}%" for density in stats['density_percent']]
    
    ax2.bar(x_labels, stats['speedup'], 
           color='skyblue', edgecolor='navy')
    ax2.set_title('Відносна продуктивність cuDSS\n(1.0 = найшвидший варіант)')
    ax2.set_xlabel('Щільність матриці')
    ax2.set_ylabel('Коефіцієнт уповільнення')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('cudss_performance_with_runs.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_phases(phase_df):
    """Аналізує час окремих етапів"""
    if phase_df is None:
        return None
    
    phases = ['func_value_t', 'jacobian_value_t', 'delta_value_t', 'update_points_t']
    
    # Додаємо кількість ненульових елементів
    phase_df['nonzeros_per_row'] = 10000 - phase_df['zeros_per_row']
    
    # Групуємо дані
    phase_stats = phase_df.groupby(['nonzeros_per_row', 'run'])[phases].mean().reset_index()
    
    # Обчислюємо статистики для кожного ступеня щільності
    result = phase_df.groupby('nonzeros_per_row')[phases].agg(['mean', 'std']).reset_index()
    result.columns = ['nonzeros_per_row'] + [f'{col[0]}_{col[1]}' for col in result.columns[1:]]
    
    # Сортуємо за зростанням ненульових елементів
    result = result.sort_values('nonzeros_per_row').reset_index(drop=True)
    
    # Побудова графіків
    plt.figure(figsize=(12, 6))
    
    for phase in phases:
        mean_col = f'{phase}_mean'
        std_col = f'{phase}_std'
        
        plt.errorbar(result['nonzeros_per_row'], result[mean_col], 
                    yerr=result[std_col], fmt='o-',
                    label=phase.replace('_t', '').replace('_', ' '),
                    capsize=5)
    
    plt.title('Час окремих етапів cuDSS залежно від кількості ненульових елементів\n')
    plt.xlabel('Кількість ненульових елементів у рядку')
    plt.ylabel('Час виконання (секунди)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cudss_phase_times_with_runs.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return result

def main():
    try:
        print("Завантаження даних з усіх запусків...")
        stats_df, phases_df = load_all_runs()
        
        print("\nАналіз продуктивності cuDSS...")
        performance_stats = analyze_cudss_performance(stats_df)
        print("\nРезультати аналізу продуктивності:")
        print(performance_stats.to_string(index=False))
        
        print("\nСтворення графіків продуктивності...")
        plot_performance(performance_stats)
        
        print("\nАналіз окремих етапів...")
        phase_stats = analyze_phases(phases_df)
        if phase_stats is not None:
            print("\nРезультати аналізу етапів:")
            print(phase_stats.to_string(index=False))
        
        print("\nЗбереження результатів...")
        performance_stats.to_csv('cudss_performance_summary.csv', index=False)
        if phase_stats is not None:
            phase_stats.to_csv('cudss_phase_stats.csv', index=False)
        
        print("\nГотово! Результати збережено у файлах:")
        print("- cudss_performance_with_runs.png")
        print("- cudss_performance_summary.csv")
        if phase_stats is not None:
            print("- cudss_phase_times_with_runs.png")
            print("- cudss_phase_stats.csv")
        
    except Exception as e:
        print(f"\nПомилка: {e}")

if __name__ == "__main__":
    main()