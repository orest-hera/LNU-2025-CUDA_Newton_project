#pragma once
#include "DataInitializer.h"
#include "mkl.h"

class DataInitializerMKLdss : public DataInitializer {
public:
	bool analyzed = false;
	double* csr_values_h{ nullptr }, * jacobian{ nullptr };
	MKL_INT* csr_rows_h{ nullptr },
		* csr_cols_h{ nullptr };

    double* funcs_value_h{ nullptr },
		* delta_h{ nullptr };

	int non_zero_count{ 0 };

    MKL_INT opt = MKL_DSS_MSG_LVL_WARNING + MKL_DSS_TERM_LVL_ERROR + MKL_DSS_ZERO_BASED_INDEXING;
    MKL_INT sym = MKL_DSS_SYMMETRIC;
    MKL_INT order = MKL_DSS_AUTO_ORDER;
    MKL_INT type = MKL_DSS_INDEFINITE;

    _MKL_DSS_HANDLE_t handle;

    int count_non_zero_elements(double* matrix_A);

    DataInitializerMKLdss(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power);
    ~DataInitializerMKLdss();
};