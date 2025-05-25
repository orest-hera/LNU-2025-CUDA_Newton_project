#pragma once
#include <DataInitializerCuDSS.h>

class NewtonSolverCuDSS {
private:
	DataInitializerCuDSS* data;
	int count_non_zero_elements(double* matrix_A);
	void parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A);
public:

	NewtonSolverCuDSS(DataInitializerCuDSS*data);
	~NewtonSolverCuDSS();
	void solve(double* matrix_A_h, double* vector_b_d, double* vector_x_h, double* vector_x_d);
	void gpu_newton_solver_cudss();
};