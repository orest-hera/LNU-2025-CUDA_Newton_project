import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./results/total_statistic.csv')

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
plt.title('Execution Time vs Matrix Size')
plt.legend()

plt.grid(True)

plt.savefig('./results/total_statistic.png')