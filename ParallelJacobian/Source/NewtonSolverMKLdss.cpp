#include "NewtonSolverMKLdss.h"

int NewtonSolverMKLdss::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < data->MATRIX_SIZE * data->MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void NewtonSolverMKLdss::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
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

void NewtonSolverMKLdss::cpu_computeVec() {
    const int N = data->MATRIX_SIZE;

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double index = data->indexes_h[i * N + j];
            double x = data->points_h[j];
            sum += data->equation->calculate_term_value(index, x);
        }

        data->funcs_value_h[i] = sum - data->vector_b_h[i];
    }
}

// double NewtonSolverMKLdss::cpu_compute_derivative(int rowIndex, int colIndex) {
//     double value = data->points_h[colIndex];
//     double element = data->indexes_h[rowIndex * data->MATRIX_SIZE + colIndex];

//     double f_plus =  data->equation->calculate_term_value(element, value + EQURENCY);
//     double f_minus = data->equation->calculate_term_value(element, value - EQURENCY);;

//     return (f_plus - f_minus) / (2.0 * EQURENCY);
// }

// void NewtonSolverMKLdss::cpu_compute_jacobian() {
//     for (int i = 0; i < data->MATRIX_SIZE; ++i) {
//         for (int j = 0; j < data->MATRIX_SIZE; ++j) {
//             data->jacobian_h[i * data->MATRIX_SIZE + j] = cpu_compute_derivative(i, j);
//         }
//     }
// }