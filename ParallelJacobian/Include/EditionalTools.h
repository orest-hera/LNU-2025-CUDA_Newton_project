#pragma once
#include "Equation.h"
#include "DataInitializer.h"

namespace tools {
	void generate_initial_indexes_matrix_and_vector_b(double* matrix, double* b, double* points, int MATRIX_SIZE, Equation* equation);
	void generate_sparse_initial_indexes_matrix_and_vector_b(
		double* matrix,
		double* b,
		double* points,
		int MATRIX_SIZE,
		Equation* equation,
		int zero_elements_per_row = 0);
	void generate_sparse_initial_indexes_matrix_and_vector_b(
		double* csr_values,
		int* csr_rows,
		int* csr_cols,
		double* b,
		double* points,
		int MATRIX_SIZE,
		Equation* equation,
		int zero_elements_per_row = 0);
	double calculate_index_xn(double index, double x);
	//void generate_sparse_initial_indexes_matrix_and_vector_b(double* matrix, double* b, int zeros_per_row, int MATRIX_SIZE);

	//
	// RESULTS PRINT
	//
	void print_solution(DataInitializer* data, int iterations_count);
	void print_intermediate_result(DataInitializer* data, int iteration_number, double error, bool isCudss = false);
}