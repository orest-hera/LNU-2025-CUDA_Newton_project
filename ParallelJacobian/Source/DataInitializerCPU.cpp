#include "DataInitializerCPU.h"

DataInitializerCPU::DataInitializerCPU(int MATRIX_SIZE, int zeros_elements_per_row, int power)
	: DataInitializer(MATRIX_SIZE, zeros_elements_per_row, power) {
	funcs_value_h = new double[MATRIX_SIZE];
	jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
	inverse_jacobian_h = new double[MATRIX_SIZE * MATRIX_SIZE];
	delta_h = new double[MATRIX_SIZE];
};

DataInitializerCPU::~DataInitializerCPU() {
	delete[] funcs_value_h;
	delete[] jacobian_h;
	delete[] inverse_jacobian_h;
	delete[] delta_h;
}