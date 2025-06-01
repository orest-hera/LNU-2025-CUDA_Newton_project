import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_delta_data(base_dir="../"):
    delta_data = []
    
    results_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('results_') and os.path.isdir(os.path.join(base_dir, d))])
    
    if not results_dirs:
        raise FileNotFoundError("Папки з результатами не знайдено (results_*)")
    
    for run_dir in results_dirs:
        run_path = os.path.join(base_dir, run_dir)
        
        for file in os.listdir(run_path):
            if file.endswith('.csv') and not file.startswith('total_statistic'):
                file_path = os.path.join(run_path, file)
                df = pd.read_csv(file_path)

                df['iteration'] = range(1, len(df)+1)
                
                if 'zeros_per_row' not in df.columns:
                    try:
                        zeros = int(file.split('_')[-1].split('.')[0])
                        df['zeros_per_row'] = zeros
                        df['nonzeros_per_row'] = 10000 - zeros
                    except:
                        continue
                
                df['run'] = run_dir
                
                if 'delta_value_t' in df.columns:
                    delta_data.append(df[['nonzeros_per_row', 'delta_value_t', 'iteration', 'run']])
    
    if not delta_data:
        raise ValueError("Дані фази дельта не знайдено в папках з результатами")
    
    return pd.concat(delta_data, ignore_index=True)

def analyze_and_plot_delta(delta_df):
    first_iter = delta_df[delta_df['iteration'] == 1]
    other_iters = delta_df[delta_df['iteration'] > 1]
    
    first_stats = first_iter.groupby('nonzeros_per_row')['delta_value_t'].agg(['mean', 'std', 'count']).reset_index()
    other_stats = other_iters.groupby('nonzeros_per_row')['delta_value_t'].agg(['mean', 'std', 'count']).reset_index()
    
    first_stats = first_stats.sort_values('nonzeros_per_row')
    other_stats = other_stats.sort_values('nonzeros_per_row')
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(first_stats['nonzeros_per_row'], first_stats['mean'], 
             'o-', color='blue', linewidth=2, markersize=8,
             label='Перша ітерація (з етапом аналізу системи)')
    
    plt.plot(other_stats['nonzeros_per_row'], other_stats['mean'], 
             's-', color='red', linewidth=2, markersize=8,
             label='Середнє значення інших ітерацій')
    
    plt.fill_between(other_stats['nonzeros_per_row'],
                     other_stats['mean'] - other_stats['std'],
                     other_stats['mean'] + other_stats['std'],
                     color='red', alpha=0.2)
    
    plt.title('Час виконання фази дельта: Перша проти інших ітерацій\n')
    plt.xlabel('Кількість ненульових елементів у рядку')
    plt.ylabel('Час виконання (секунди)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('delta_phase_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return first_stats, other_stats

def main():
    try:
        print("Завантаження даних фази дельта з усіх запусків...") 
        delta_df = load_delta_data()
        
        print("\nАналіз та побудова графіка порівняння фази дельта...")
        first_stats, other_stats = analyze_and_plot_delta(delta_df)
        
        print("\nСтатистика фази дельта:") 
        print("\nПерша ітерація:")
        print(first_stats.to_string(index=False))
        print("\nІнші ітерації:")
        print(other_stats.to_string(index=False))
        
        print("\nЗбереження результатів...")
        first_stats.to_csv('delta_first_iteration_stats.csv', index=False)
        other_stats.to_csv('delta_other_iterations_stats.csv', index=False)
        
        print("\nГотово! Результати збережено в:") 
        print("- delta_phase_comparison.png")
        print("- delta_first_iteration_stats.csv")
        print("- delta_other_iterations_stats.csv")
        
    except Exception as e:
        print(f"\nПомилка: {e}")

if __name__ == "__main__":
    main()