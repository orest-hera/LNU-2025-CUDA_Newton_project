# CUDA Newton Matrix Solver Benchmark

This project benchmarks the performance of solving systems of linear equations using NVIDIA GPUs. It supports both dense and sparse matrix formats and provides automation tools for repeated testing and result visualization.

## 🧩 Requirements

To use this program, you must have a computer equipped with a discrete NVIDIA GPU.  
This project was tested using a **Turing-based GPU** (Compute Capability 7.X).

### ✅ Required Software

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDSS (CUDA Direct Sparse Solver)](https://developer.nvidia.com/cuda-toolkit)

> ⚠️ **Installation Note:**  
> It is recommended to install the libraries using the default paths suggested by the installer.  
> If you choose custom paths, you must manually set them in the `CMakeLists.txt` file.

Even if the default paths do not work, don't worry — the `CMakeLists.txt` file includes dynamic detection logic for finding installed libraries.

---

## ⚙️ Building the Project

1. Open PowerShell or your terminal.
2. Navigate to the `scripts` directory.
3. Run the build script:

```powershell
./build.ps1
```

  After a successful build, the program is ready to use.

---

## 🚀 Running the Program

There are three modes for running the benchmark:

1. Dense Matrix Mode
   Use this mode to benchmark using dense matrices:
```powershell
./run_dimention.ps1 <max_size> <min_size> <step>
```

2. Sparse Matrix Mode
   Use this mode to benchmark using sparse matrices:
```powershell
./run_sparse.ps1 <max_zeros> <min_zeros> <step>
```

3. Repeated Run Mode
   Use this mode to run multiple benchmarks for better averaging:
 ```powershell
./run_in_loop.ps1 <max> <min> <step>
```

### Example:
 ```powershell
./run_in_loop.ps1 1000 100 100
```
---

## 📁 Output

After running a script, the program will create:
- A results/ folder in the root directory, or
- Multiple folders (e.g., `results_1`, `results_2`, `etc`.) if using loop mode

These folders contain `.csv` files with benchmark data (execution time, matrix size, GPU details, etc.).

---

## 📊 Visualizing Results

To generate plots from the .csv data, run the provided Python script:

 ```powershell
python draw.py
```
The graphs will help you analyze how performance scales with matrix size or sparsity.

---

## 🧾 Example Workflow

```powershell
# Build the project
cd scripts
./build.ps1

# Run the benchmark (looped test with dense matrices)
./run_in_loop.ps1 1000 100 100

# Visualize results
python draw.py
```
