#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace NewtonSolverFunctions {
	__global__ void gpu_dummy_warmup();
	__global__ void gpu_compute_jacobian(double* points_d, double* indexes_d, double* jacobian_d, int MATRIX_SIZE, int power);
	__global__ void gpu_compute_delta_values(double* points_d, double* indexes_d, double* vec_d, int MATRIX_SIZE, int version);
	__global__ void gpu_compute_func_values(double* points_d, double* indexes_d, double* vec_d, int MATRIX_SIZE, int version, int power);
}
