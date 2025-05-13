#pragma once
#include "DataInitializer.h"
#include <cuda.h>

class CuDssSolver {
private:
	DataInitializer* data;
	int count_non_zero_elements(double* matrix_A);
	void parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A);
public:
	double * csr_values_h{ nullptr };

	int* csr_rows_h{ nullptr },
		* csr_cols_h{ nullptr };

	int* csr_rows_d{ nullptr },
		* csr_cols_d{ nullptr };

	double* csr_values_d{ nullptr };

	int non_zero_count{ 0 };

	CuDssSolver(DataInitializer *data);
	~CuDssSolver();
	double solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d);
	void gpu_newton_solver_cudss();
	void print_solutionn(int iterations_count, double* result, double* initial, DataInitializer* data);
};