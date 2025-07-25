#include "DataInitializerMKLdss.h"
#include <iostream>

int DataInitializerMKLdss::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

DataInitializerMKLdss::DataInitializerMKLdss(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power)
    : DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, power){
	// Allocate memory for CSR representation
	non_zero_count = count_non_zero_elements(indexes_h);
	csr_values_h = new double[non_zero_count];
	jacobian = new double[non_zero_count];
	csr_rows_h = new int[MATRIX_SIZE + 1];
	csr_cols_h = new int[non_zero_count];

    delta_h = new double[MATRIX_SIZE];
	funcs_value_h = new double[MATRIX_SIZE];

    MKL_INT status = dss_create(handle, opt);
	if (status != MKL_DSS_SUCCESS) {
		std::cout << "Error in dss_create: " << status << std::endl;
    }
}

DataInitializerMKLdss::~DataInitializerMKLdss(){
    delete[] csr_values_h;
	delete[] jacobian;
    delete[] csr_cols_h;
    delete[] csr_rows_h;

	delete[] delta_h;
	delete[] funcs_value_h;

    dss_delete(handle, opt);
}