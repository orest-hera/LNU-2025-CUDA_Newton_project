#include "NewtonSolverCuDSS.h"
#include "NewtonSolverGPUFunctions.h"
#include "DataInitializer.h"
#include "FileOperations.h"
#include "iostream"
#include "math.h"
#include "chrono"
#include "cudss.h"

NewtonSolverCuDSS::NewtonSolverCuDSS(DataInitializer* data) {
	this->data = data;

	non_zero_count = count_non_zero_elements(data->indexes_h);
	csr_cols_h = new int[non_zero_count];
	csr_rows_h = new int[data->MATRIX_SIZE + 1];
	csr_values_h = new double[non_zero_count];

	cudaMalloc((void**)&csr_values_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&csr_rows_d, (data->MATRIX_SIZE + 1) * sizeof(int));
	cudaMalloc((void**)&csr_cols_d, non_zero_count * sizeof(int));
}

NewtonSolverCuDSS::~NewtonSolverCuDSS() {
	delete[] csr_cols_h;
	delete[] csr_rows_h;
	delete[] csr_values_h;
	cudaFree(csr_values_d);
	cudaFree(csr_rows_d);
	cudaFree(csr_cols_d);
}

int NewtonSolverCuDSS::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < data->MATRIX_SIZE * data->MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void NewtonSolverCuDSS::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
	int non_zero_count = 0;
	csr_rows[0] = 0;
	for (int i = 0; i < data->MATRIX_SIZE; ++i) {
		for (int j = 0; j < data->MATRIX_SIZE; ++j) {
			if (matrix_A[i * data->MATRIX_SIZE + j] != 0) {
				csr_cols[non_zero_count] = j;
				csr_values[non_zero_count] = matrix_A[i * data->MATRIX_SIZE + j];
				non_zero_count++;
			}
		}
		csr_rows[i + 1] = non_zero_count;
	}
}

double NewtonSolverCuDSS::solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d) {
	parse_to_csr(csr_cols_h, csr_rows_h, csr_values_h, matrix_A_h);

	cudaMemcpy(csr_cols_d, csr_cols_h, non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_rows_d, csr_rows_h, (data->MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_values_d, csr_values_h, non_zero_count * sizeof(double), cudaMemcpyHostToDevice);

	auto start = std::chrono::high_resolution_clock::now();
	cudssHandle_t handler;
	cudssConfig_t solverConfig;
	cudssData_t solverData;
	cudssCreate(&handler);

	cudssConfigCreate(&solverConfig);
	cudssDataCreate(handler, &solverData);

	cudssMatrix_t x, b;
	cudssMatrixCreateDn(&b, data->MATRIX_SIZE, 1, data->MATRIX_SIZE, vector_b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
	cudssMatrixCreateDn(&x, data->MATRIX_SIZE, 1, data->MATRIX_SIZE, vector_x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

	cudssMatrix_t A;
	cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
	cudssMatrixViewType_t mvtype = CUDSS_MVIEW_FULL;
	cudssIndexBase_t base = CUDSS_BASE_ZERO;
	cudssMatrixCreateCsr(&A, data->MATRIX_SIZE, data->MATRIX_SIZE, non_zero_count, csr_rows_d, NULL, csr_cols_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mvtype, base);

	cudssExecute(handler, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, x, b);
	cudssExecute(handler, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, x, b);
	cudssExecute(handler, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, x, b);

	cudssMatrixDestroy(A);
	cudssMatrixDestroy(x);
	cudssMatrixDestroy(b);
	cudssDataDestroy(handler, solverData);
	cudssConfigDestroy(solverConfig);
	cudssDestroy(handler);

	cudaDeviceSynchronize();

	cudaMemcpy(vector_x_h, vector_x_d, data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
	return elapsed.count();
}

void NewtonSolverCuDSS::print_solutionn(int iterations_count, double* result, double* initial, DataInitializer* data) {
	std::cout << "Total Iterations count: " << iterations_count << "\n";
#ifdef SOLUTION_PRINT
	std::cout << "Solution: \n";

	for (int i = 0; i < data->MATRIX_SIZE; i++) {
		std::cout << result[i] << " " << initial[i] << "\n";
	}
#endif
	std::cout << "\n";
}

void NewtonSolverCuDSS::gpu_newton_solver_cudss() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int version = prop.major;
	FileOperations* file_op = new FileOperations();
	std::string file_name = "gpu_cudss_newton_solver_" + std::to_string(data->MATRIX_SIZE) + ".csv";
	file_op->create_file(file_name, 5);
	file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");

	NewtonSolverGPUFunctions::gpu_dummy_warmup << <1, 32 >> > ();
	cudaDeviceSynchronize();
	std::cout << "GPU CuDss Newton solver\n";
	int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int iterations_count = 0;
	double dx = 0;

	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

	auto start_total = std::chrono::high_resolution_clock::now();

	cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
		data->points_h[i] += data->delta_h[i];
		dx = std::max(dx, std::abs(data->delta_h[i]));
	}

	do {
		iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::high_resolution_clock::now();
#endif
		std::cout << "Power: " << data->equation->get_power() << "\n";
		NewtonSolverGPUFunctions::gpu_compute_func_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
			data->points_d, data->indexes_d, data->intermediate_funcs_value_d, data->MATRIX_SIZE, version, data->equation->get_power());
		cudaDeviceSynchronize();

		cudaMemcpy(data->intermediate_funcs_value_h, data->intermediate_funcs_value_d, x_blocks_count * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

		for (int i = 0; i < data->MATRIX_SIZE; i++) {
			data->funcs_value_h[i] = -data->vector_b_h[i];
			for (int j = 0; j < x_blocks_count; j++) {
				data->funcs_value_h[i] += data->intermediate_funcs_value_h[i * x_blocks_count + j];
			}
		}
		cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::high_resolution_clock::now();
#endif

		NewtonSolverGPUFunctions::gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
			data->points_d, data->indexes_d, data->jacobian_d, data->MATRIX_SIZE, data->equation->get_power());
		cudaDeviceSynchronize();

		cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::high_resolution_clock::now();
#endif
		for (int i = 0; i < data->MATRIX_SIZE; i++) {
			data->funcs_value_h[i] *= -1;
		}
		cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

		solve(data->jacobian_h, data->funcs_value_d, data->delta_h, data->delta_d);
		dx = 0.0;
		for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] += data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}
		std::cout << "dx: " << dx << "\n";
		cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration<double>(end_total - start_total).count();
	std::cout << "Total time: " << total_time << "\n";
	print_solutionn(iterations_count, data->points_h, data->points_check, data);
}