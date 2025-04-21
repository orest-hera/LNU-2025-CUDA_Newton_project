import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

results_folder = '../results/'

cpu_files = sorted(glob.glob(os.path.join(results_folder, 'cpu_newton_solver_*.csv')))
gpu_files = sorted(glob.glob(os.path.join(results_folder, 'gpu_newton_solver_*.csv')))

stages = ['func_value_t', 'jacobian_value_t', 'inverse_jacobian_t', 'delta_value_t']

matrix_sizes = []
cpu_stage_values = {stage: [] for stage in stages}
gpu_stage_values = {stage: [] for stage in stages}

for cpu_file, gpu_file in zip(cpu_files, gpu_files):
    cpu_data = pd.read_csv(cpu_file)
    gpu_data = pd.read_csv(gpu_file)

    matrix_size = cpu_data['matrix_size'].iloc[0]
    matrix_sizes.append(matrix_size)

    for stage in stages:
        cpu_stage_values[stage].append(cpu_data[stage].mean())
        gpu_stage_values[stage].append(gpu_data[stage].mean())

for stage in stages:
    plt.figure(figsize=(10, 6))

    plt.plot(matrix_sizes, cpu_stage_values[stage], marker='o', label='CPU')
    plt.plot(matrix_sizes, gpu_stage_values[stage], marker='s', label='GPU')

    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (s)')
    plt.title(f'Execution Time vs Matrix Size ({stage})')
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(results_folder, f'{stage}_compare.png')
    plt.savefig(output_path)
    plt.close()

data = pd.read_csv(os.path.join(results_folder, 'total_statistic.csv'))

matrix_sizes = data['matrix_size']
cpu_times = data['CPU']
gpu_times = data['GPU']
cudss_times = data['cuDSS']

plt.figure(figsize=(10, 6))

plt.plot(matrix_sizes, cpu_times, marker='o', label='CPU')
plt.plot(matrix_sizes, gpu_times, marker='s', label='GPU')
plt.plot(matrix_sizes, cudss_times, marker='^', label='cuDSS')

plt.xlabel('Matrix Size')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time vs Matrix Size (Total)')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(results_folder, 'total_statistic.png'))
plt.close()