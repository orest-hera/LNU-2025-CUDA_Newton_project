#pragma once
#include "DataInitializer.h"
#include "mkl_dss.h"
#include "mkl_types.h"

class DataInitializerMKLdss : public DataInitializer {
public:
	bool analyzed = false;
	double* csr_values_h{ nullptr };
	int* csr_rows_h{ nullptr },
		* csr_cols_h{ nullptr };

    double* funcs_value_h{ nullptr },
		* delta_h{ nullptr };

	int non_zero_count{ 0 };

    MKL_INT opt = MKL_DSS_DEFAULTS;
    MKL_INT sym = MKL_DSS_SYMMETRIC;
    MKL_INT type = MKL_DSS_POSITIVE_DEFINITE;

    _MKL_DSS_HANDLE_t handle;

    int count_non_zero_elements(double* matrix_A);

    DataInitializerMKLdss(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power);
    ~DataInitializerMKLdss();
};