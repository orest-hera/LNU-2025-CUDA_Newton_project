#include "cudss.h"
#include <iostream>
#include "CuDssSolver.h"
#include "EditionalTools.h"
#include "DataInitializer.h"

CuDssSolver::CuDssSolver() {
	matrix_A = new double[MATRIX_SIZE * MATRIX_SIZE];
	vector_b_h = new double[MATRIX_SIZE];

	tools::generate_initial_indexes_matrix_and_vector_b(matrix_A, vector_b_h);

	non_zero_count = count_non_zero_elements(matrix_A);
	csr_cols_h = new int[non_zero_count];
	csr_rows_h = new int[MATRIX_SIZE + 1];
	csr_values_h = new double[non_zero_count];
	vector_x_h = new double[MATRIX_SIZE];

	cudaMalloc((void**)&vector_x_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&vector_b_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&csr_values_d, non_zero_count * sizeof(double));
	cudaMalloc((void**)&csr_rows_d, (MATRIX_SIZE + 1) * sizeof(int));
	cudaMalloc((void**)&csr_cols_d, non_zero_count * sizeof(int));

	parse_to_csr(csr_cols_h, csr_rows_h, csr_values_h, matrix_A);

	cudaMemcpy(csr_cols_d, csr_cols_h, non_zero_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_rows_d, csr_rows_h, (MATRIX_SIZE + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(csr_values_d, csr_values_h, non_zero_count * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(vector_b_d, vector_b_h, MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
}

CuDssSolver::~CuDssSolver() {
	delete[] matrix_A;
	delete[] vector_b_h;
	delete[] csr_cols_h;
	delete[] csr_rows_h;
	delete[] csr_values_h;
	delete[] vector_x_h;
	cudaFree(vector_x_d);
	cudaFree(vector_b_d);
	cudaFree(csr_values_d);
	cudaFree(csr_rows_d);
	cudaFree(csr_cols_d);
}

int CuDssSolver::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void CuDssSolver::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
	int non_zero_count = 0;
	csr_rows[0] = 0;
	for (int i = 0; i < MATRIX_SIZE; ++i) {
		for (int j = 0; j < MATRIX_SIZE; ++j) {
			if (matrix_A[i * MATRIX_SIZE + j] != 0) {
				csr_cols[non_zero_count] = j;
				csr_values[non_zero_count] = matrix_A[i * MATRIX_SIZE + j];
				non_zero_count++;
			}
		}
		csr_rows[i + 1] = non_zero_count;
	}
}

void CuDssSolver::solve(){
	cudssHandle_t handler;
	cudssConfig_t solverConfig;
	cudssData_t solverData;
	cudssCreate(&handler);

	cudssConfigCreate(&solverConfig);
	cudssDataCreate(handler, &solverData);

	cudssMatrix_t x, b;
	cudssMatrixCreateDn(&b, MATRIX_SIZE, 1, MATRIX_SIZE, vector_b_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);
	cudssMatrixCreateDn(&x, MATRIX_SIZE, 1, MATRIX_SIZE, vector_x_d, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR);

	cudssMatrix_t A;
	cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
	cudssMatrixViewType_t mvtype = CUDSS_MVIEW_FULL;
	cudssIndexBase_t base = CUDSS_BASE_ZERO;
	cudssMatrixCreateCsr(&A, MATRIX_SIZE, MATRIX_SIZE, non_zero_count, csr_rows_d, NULL, csr_cols_d, csr_values_d, CUDA_R_32I, CUDA_R_64F, mtype, mvtype, base);

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

	cudaMemcpy(vector_x_h, vector_x_d, MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < MATRIX_SIZE; ++i) {
		std::cout << vector_x_h[i] << std::endl;
	}
}