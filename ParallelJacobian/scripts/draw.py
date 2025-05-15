import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

results_folder = '../results/'
images_folder = os.path.join(results_folder, 'images')

os.makedirs(images_folder, exist_ok=True)

cpu_files = sorted(glob.glob(os.path.join(results_folder, 'cpu_newton_solver_*.csv')))
gpu_files = sorted(glob.glob(os.path.join(results_folder, 'gpu_newton_solver_*.csv')))
cudss_files = sorted(glob.glob(os.path.join(results_folder, 'gpu_cudss_newton_solver_*.csv')))

all_stages = ['func_value_t', 'jacobian_value_t', 'inverse_jacobian_t', 'delta_value_t', 'update_points_t']
cudss_stages = ['func_value_t', 'jacobian_value_t', 'delta_value_t', 'update_points_t']

matrix_sizes = []

cpu_stage_values = {stage: [] for stage in all_stages}
gpu_stage_values = {stage: [] for stage in all_stages}
cudss_stage_values = {stage: [] for stage in cudss_stages}

for cpu_file, gpu_file, cudss_file in zip(cpu_files, gpu_files, cudss_files):
    cpu_data = pd.read_csv(cpu_file)
    gpu_data = pd.read_csv(gpu_file)
    cudss_data = pd.read_csv(cudss_file)

    matrix_size = int(cpu_data['matrix_size'].iloc[0])
    matrix_sizes.append(matrix_size)

    for stage in all_stages:
        cpu_stage_values[stage].append(cpu_data[stage].mean() if stage in cpu_data.columns else 0)
        gpu_stage_values[stage].append(gpu_data[stage].mean() if stage in gpu_data.columns else 0)

    for stage in cudss_stages:
        cudss_stage_values[stage].append(cudss_data[stage].mean() if stage in cudss_data.columns else 0)

matrix_sizes_arr = np.array(matrix_sizes)
sorted_indices = np.argsort(matrix_sizes_arr)
matrix_sizes_sorted = matrix_sizes_arr[sorted_indices]

def sort_stage_values(stage_values):
    return np.array(stage_values)[sorted_indices]

cpu_stage_values_sorted = {stage: sort_stage_values(cpu_stage_values[stage]) for stage in all_stages}
gpu_stage_values_sorted = {stage: sort_stage_values(gpu_stage_values[stage]) for stage in all_stages}
cudss_stage_values_sorted = {stage: sort_stage_values(cudss_stage_values[stage]) for stage in cudss_stages}

for stage in all_stages:
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes_sorted, cpu_stage_values_sorted[stage], marker='o', label='CPU')
    plt.plot(matrix_sizes_sorted, gpu_stage_values_sorted[stage], marker='s', label='GPU')
    if stage in cudss_stages:
        plt.plot(matrix_sizes_sorted, cudss_stage_values_sorted[stage], marker='^', label='cuDSS')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs Matrix Size ({stage})')
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(images_folder, f'{stage}_compare.png')
    plt.savefig(output_path)
    plt.close()

data = pd.read_csv(os.path.join(results_folder, 'total_statistic.csv'))

matrix_sizes_total = data['matrix_size']
cpu_times = data['CPU']
gpu_times = data['GPU']
cudss_times = data['cuDSS']

total_sorted_indices = np.argsort(matrix_sizes_total)
matrix_sizes_total_sorted = matrix_sizes_total.iloc[total_sorted_indices]
cpu_times_sorted = cpu_times.iloc[total_sorted_indices]
gpu_times_sorted = gpu_times.iloc[total_sorted_indices]
cudss_times_sorted = cudss_times.iloc[total_sorted_indices]

plt.figure(figsize=(10, 6))
plt.plot(matrix_sizes_total_sorted, cpu_times_sorted, marker='o', label='CPU')
plt.plot(matrix_sizes_total_sorted, gpu_times_sorted, marker='s', label='GPU')
plt.plot(matrix_sizes_total_sorted, cudss_times_sorted, marker='^', label='cuDSS')

plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Matrix Size (Total)')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(images_folder, 'total_statistic.png'))
plt.close()