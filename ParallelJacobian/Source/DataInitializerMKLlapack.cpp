#include "DataInitializerMKLlapack.h"

DataInitializerMKLlapack::DataInitializerMKLlapack(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power)
	: DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, power) {
	funcs_value_h = new double[MATRIX_SIZE];
	jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
};

DataInitializerMKLlapack::~DataInitializerMKLlapack() {
	delete[] funcs_value_h;
	delete[] jacobian_h;
}