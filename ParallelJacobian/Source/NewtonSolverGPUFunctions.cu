#include "NewtonSolverGPUFunctions.h"
#include "DataInitializer.h"

__global__ void NewtonSolverGPUFunctions::gpu_dummy_warmup() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < 32) {
        double tmp = idx * 0.1;
    }
}

__global__ void NewtonSolverGPUFunctions::gpu_compute_func_values(double* points_d, double* indexes_d, double* vec_d, int MATRIX_SIZE, int version, int power) {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x;

    extern __shared__ double shared_points[];

    if (gidx < MATRIX_SIZE) {
        double value = 1.0;
        for (int i = 0; i < power; i++) {
            value *= points_d[gidx];
        }
        shared_points[threadIdx.x] = value * indexes_d[gidy * MATRIX_SIZE + gidx];
    }
    else {
        shared_points[threadIdx.x] = 0.0;
    }
    __syncthreads();


    if (BLOCK_SIZE >= 1024 && threadIdx.x < 512) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 512];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 512 && threadIdx.x < 256) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 256];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 256 && threadIdx.x < 128) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 128];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 128 && threadIdx.x < 64) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 64];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 64 && threadIdx.x < 32) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 32];
    }

    __syncthreads();

    if (threadIdx.x < 32) {

        if (version >= 7) {
            double sum = shared_points[threadIdx.x];
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 16);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 8);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 4);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 2);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 1);
            if (threadIdx.x == 0) {
                vec_d[gidy * x_blocks_count + blockIdx.x] = sum;
            }
        }
        else {
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 16]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 8]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 4]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 2]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 1]; __syncwarp();
            if (tidx == 0) {
                vec_d[gidy * x_blocks_count + blockIdx.x] = shared_points[threadIdx.x];
            }
        }
    }
}

__global__ void NewtonSolverGPUFunctions::gpu_compute_delta_values(double* points_d, double* indexes_d, double* vec_d, int MATRIX_SIZE, int version) {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    int gidy = blockDim.y * blockIdx.y + threadIdx.y;
    int tidx = threadIdx.x;

    extern __shared__ double shared_points[];

    if (gidx < MATRIX_SIZE) {
        shared_points[threadIdx.x] = points_d[gidx] * indexes_d[gidy * MATRIX_SIZE + gidx];
    }
    else {
        shared_points[threadIdx.x] = 0.0;
    }
    __syncthreads();


    if (BLOCK_SIZE >= 1024 && threadIdx.x < 512) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 512];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 512 && threadIdx.x < 256) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 256];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 256 && threadIdx.x < 128) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 128];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 128 && threadIdx.x < 64) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 64];
    }

    __syncthreads();

    if (BLOCK_SIZE >= 64 && threadIdx.x < 32) {
        shared_points[threadIdx.x] += shared_points[threadIdx.x + 32];
    }

    __syncthreads();

    if (threadIdx.x < 32) {

        if (version >= 7) {
            double sum = shared_points[threadIdx.x];
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 16);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 8);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 4);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 2);
            sum += __shfl_down_sync(SHAFFLE_CONST, sum, 1);
            if (threadIdx.x == 0) {
                vec_d[gidy * x_blocks_count + blockIdx.x] = sum;
            }
        }
        else {
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 16]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 8]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 4]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 2]; __syncwarp();
            shared_points[threadIdx.x] += shared_points[threadIdx.x + 1]; __syncwarp();
            if (tidx == 0) {
                vec_d[gidy * x_blocks_count + blockIdx.x] = shared_points[threadIdx.x];
            }
        }
    }
}

__global__ void NewtonSolverGPUFunctions::gpu_compute_jacobian(double* points_d, double* indexes_d, double* jacobian_d, int MATRIX_SIZE, int power) {
    extern __shared__ double shared_data[];

    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    double result = 0.0;
    double f_minus = 0.0;
    double f_plus = 0.0;

    for (int ph = 0; ph < gridDim.x; ++ph) {
        int global_col = ph * blockDim.x + threadIdx.x;

        if (global_col < MATRIX_SIZE) {
            shared_data[threadIdx.x] = points_d[global_col];
            shared_data[blockDim.x + threadIdx.x] = indexes_d[row * MATRIX_SIZE + global_col];
        }
        else {
            shared_data[threadIdx.x] = 0.0;
            shared_data[blockDim.x + threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int i = 0; i < blockDim.x; ++i) {
            if (ph * blockDim.x + i >= MATRIX_SIZE) break;

            double value = shared_data[i];
            double element = shared_data[blockDim.x + i];

            if (ph * blockDim.x + i == col) {
                double x_value_plus = 1;
                double x_value_minus = 1;
                for (int i = 0; i < power; i++) {
                    x_value_plus *= (value + EQURENCY);
                    x_value_minus *= (value - EQURENCY);
                }
                f_minus += x_value_minus * element;
                f_plus += x_value_plus * element;
            }
            else {
                f_minus += value * element;
                f_plus += value * element;
            }
        }

        __syncthreads();
    }

    result = (f_plus - f_minus) / (2 * EQURENCY);

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        jacobian_d[row * MATRIX_SIZE + col] = result;
    }
}