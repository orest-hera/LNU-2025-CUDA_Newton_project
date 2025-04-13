#pragma once

class CuDssSolver {
private:
	int count_non_zero_elements(double* matrix_A);
	void parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A);
public:
	double* vector_b_h{ nullptr },
		* vector_x_h{ nullptr },
		* matrix_A{ nullptr },
		* csr_values_h{ nullptr };

	int* csr_rows_h{ nullptr },
		* csr_cols_h{ nullptr };

	int* csr_rows_d{ nullptr },
		* csr_cols_d{ nullptr };

	double* csr_values_d{ nullptr },
		* vector_x_d{ nullptr },
		* vector_b_d{ nullptr };

	int non_zero_count{ 0 };

	CuDssSolver();
	//~CuDssSolver();
	void solve();
};