import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_gpu_data(base_dir="../"):
    """Load GPU timing data from all results folders"""
    cublas_data = []
    cudss_data = []
    
    # Find all results folders
    results_dirs = [d for d in os.listdir(base_dir) if d.startswith('results_') and os.path.isdir(os.path.join(base_dir, d))]
    
    if not results_dirs:
        raise FileNotFoundError("No results folders found (results_*)")
    
    for run_dir in results_dirs:
        run_path = os.path.join(base_dir, run_dir)
        
        # Read all CSV files in the folder
        for file in os.listdir(run_path):
            if not file.endswith('.csv'):
                continue
                
            file_path = os.path.join(run_path, file)
            
            # Process cublas files (gpu_newton_solver_*)
            if 'gpu_newton_solver_' in file:
                df = pd.read_csv(file_path)
                if 'inverse_jacobian_t' in df.columns:
                    # Add matrix size from filename
                    try:
                        matrix_size = int(file.split('_')[-1].split('.')[0])
                        df['matrix_size'] = matrix_size
                        df['type'] = 'cublas'
                        df['iteration'] = range(1, len(df)+1)
                        cublas_data.append(df[['matrix_size', 'iteration', 'inverse_jacobian_t', 'type']])
                    except:
                        continue
            
            # Process cudss files (gpu_cudss_newton_solver_*)
            elif 'gpu_cudss_newton_solver_' in file:
                df = pd.read_csv(file_path)
                if 'delta_value_t' in df.columns:
                    # Add matrix size from filename
                    try:
                        matrix_size = int(file.split('_')[-1].split('.')[0])
                        df['matrix_size'] = matrix_size
                        df['type'] = 'cudss'
                        df['iteration'] = range(1, len(df)+1)
                        cudss_data.append(df[['matrix_size', 'iteration', 'delta_value_t', 'type']])
                    except:
                        continue
    
    if not cublas_data and not cudss_data:
        raise ValueError("No GPU timing data found in results folders")
    
    cublas_df = pd.concat(cublas_data, ignore_index=True) if cublas_data else pd.DataFrame()
    cudss_df = pd.concat(cudss_data, ignore_index=True) if cudss_data else pd.DataFrame()
    
    return cublas_df, cudss_df

def analyze_and_plot_gpu(cublas_df, cudss_df):
    """Analyze and plot GPU phase comparison"""
    # Prepare figure
    plt.figure(figsize=(12, 6))
    
    # Plot cublas inverse_jacobian_t
    if not cublas_df.empty:
        cublas_stats = cublas_df.groupby('matrix_size')['inverse_jacobian_t'].agg(['mean', 'std', 'count'])
        plt.errorbar(cublas_stats.index, cublas_stats['mean'], 
                    yerr=cublas_stats['std'], fmt='o-', 
                    color='blue', linewidth=2, markersize=8,
                    label='cuBLAS (inverse_jacobian_t)',
                    capsize=5)
    
    # Plot cudss delta_value_t
    if not cudss_df.empty:
        cudss_stats = cudss_df.groupby('matrix_size')['delta_value_t'].agg(['mean', 'std', 'count'])
        plt.errorbar(cudss_stats.index, cudss_stats['mean'], 
                    yerr=cudss_stats['std'], fmt='s-', 
                    color='red', linewidth=2, markersize=8,
                    label='cuDSS (delta_value_t)',
                    capsize=5)
    
    plt.title('GPU Performance Comparison: cuBLAS vs cuDSS\n')
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig('gpu_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return cublas_stats if not cublas_df.empty else None, cudss_stats if not cudss_df.empty else None

def main():
    try:
        print("Loading GPU timing data from all runs...")
        cublas_df, cudss_df = load_gpu_data()
        
        print("\nAnalyzing and plotting GPU performance comparison...")
        cublas_stats, cudss_stats = analyze_and_plot_gpu(cublas_df, cudss_df)
        
        print("\nGPU timing statistics:")
        if cublas_stats is not None:
            print("\ncuBLAS inverse_jacobian_t:")
            print(cublas_stats.to_string())
        if cudss_stats is not None:
            print("\ncuDSS delta_value_t:")
            print(cudss_stats.to_string())
        
        print("\nSaving results...")
        if cublas_stats is not None:
            cublas_stats.to_csv('cublas_inverse_jacobian_stats.csv')
        if cudss_stats is not None:
            cudss_stats.to_csv('cudss_delta_stats.csv')
        
        print("\nDone! Results saved in:")
        print("- gpu_performance_comparison.png")
        if cublas_stats is not None:
            print("- cublas_inverse_jacobian_stats.csv")
        if cudss_stats is not None:
            print("- cudss_delta_stats.csv")
        
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()