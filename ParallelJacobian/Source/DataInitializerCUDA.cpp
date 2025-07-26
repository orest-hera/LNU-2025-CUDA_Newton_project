#include "DataInitializerCUDA.h"

DataInitializerCUDA::DataInitializerCUDA(int MATRIX_SIZE, int zeros_elements_per_row, int file_name, int power = 1) : DataInitializer(MATRIX_SIZE, zeros_elements_per_row, file_name, power) {
	cublasCreate_v2(&cublasContextHandler);

	int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaMalloc((void**)&points_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&indexes_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&intermediate_funcs_value_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&inverse_jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&delta_d, x_blocks_count * MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&funcs_value_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&vector_b_d, MATRIX_SIZE * sizeof(double));
	cudaMalloc((void**)&cublas_pivot, MATRIX_SIZE * sizeof(int));
	cudaMalloc((void**)&cublas_info, sizeof(int));
	cudaMalloc((void**)&cublas_ajacobian_d, sizeof(double*));
	cudaMalloc((void**)&cublas_afunc_values_d, sizeof(double*));
	cudaMemcpy(cublas_ajacobian_d, &jacobian_d, sizeof(double*), cudaMemcpyHostToDevice);
	cudaMemcpy(cublas_afunc_values_d, &funcs_value_d, sizeof(double*), cudaMemcpyHostToDevice);

	intermediate_funcs_value_h = new double[x_blocks_count * MATRIX_SIZE];
	delta_h = new double[x_blocks_count * MATRIX_SIZE];
	funcs_value_h = new double[MATRIX_SIZE];
}

DataInitializerCUDA::~DataInitializerCUDA() {
	cublasDestroy_v2(cublasContextHandler);
	cudaFree(points_d);
	cudaFree(indexes_d);
	cudaFree(intermediate_funcs_value_d);
	cudaFree(jacobian_d);
	cudaFree(inverse_jacobian_d);
	cudaFree(delta_d);
	cudaFree(funcs_value_d);
	cudaFree(vector_b_d);
	cudaFree(cublas_pivot);
	cudaFree(cublas_info);
	cudaFree(cublas_ajacobian_d);
	cudaFree(cublas_afunc_values_d);
	delete[] intermediate_funcs_value_h;
	delete[] delta_h;
	delete[] funcs_value_h;
}