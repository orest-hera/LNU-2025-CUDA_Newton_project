#include "CuDssSolver.h"
#include "EditionalTools.h"
#include "cudss.h"
#include "chrono"
#include <iostream>
#include "FileOperations.h"
#include "NewtonSolverFunctions.h"
#include <NewtonSolver.h>

CuDssSolver::CuDssSolver(DataInitializer* data) {
	this->data = data;

	//tools::generate_initial_indexes_matrix_and_vector_b(data->indexes_h, data->vector_b_h, data->points_h, data->MATRIX_SIZE, data->equation);

	non_zero_count = count_non_zero_elements(data->indexes_h);
	csr_cols_h = new int[non_zero_count];
	csr_rows_h = new int[data->MATRIX_SIZE + 1];
	csr_values_h = new double[non_zero_count];

	cudaMalloc((void**)&csr_values_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&csr_rows_d, (data->MATRIX_SIZE + 1) * sizeof(int));
	cudaMalloc((void**)&csr_cols_d, non_zero_count * sizeof(int));
}

CuDssSolver::~CuDssSolver() {
	delete[] csr_cols_h;
	delete[] csr_rows_h;
	delete[] csr_values_h;
	cudaFree(csr_values_d);
	cudaFree(csr_rows_d);
	cudaFree(csr_cols_d);
}

int CuDssSolver::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < data->MATRIX_SIZE * data->MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void CuDssSolver::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
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

double CuDssSolver::solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d) {
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

void CuDssSolver::print_solutionn(int iterations_count, double* result, double* initial, DataInitializer* data) {
	std::cout << "Total Iterations count: " << iterations_count << "\n";
#ifdef SOLUTION_PRINT
	std::cout << "Solution: \n";

	for (int i = 0; i < data->MATRIX_SIZE; i++) {
		std::cout << result[i] << " " << initial[i] << "\n";
	}
#endif
	std::cout << "\n";
}