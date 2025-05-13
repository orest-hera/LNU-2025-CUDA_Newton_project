#include "../Include/NewtonSolver.h"
#include "stdio.h"
#include <iostream>
#include <string>
#include "cuda_runtime.h"
#include "FileOperations.h"
#include "../Include/NewtonSolverFunctions.h"

__global__ void normalizeRow(double* jacobian, double* inverse, int i, double pivot, int MATRIX_SIZE) {
    int j = threadIdx.x + blockDim.x * blockIdx.x;
    if (j < MATRIX_SIZE) {
        jacobian[i * MATRIX_SIZE + j] /= pivot;
        inverse[i * MATRIX_SIZE + j] /= pivot;
    }
}

__global__ void eliminateColumn(double* jacobian, double* inverse, int i, int MATRIX_SIZE) {
    int k = blockIdx.x;
    int j = threadIdx.x;

    if (k != i && j < MATRIX_SIZE) {
        double factor = jacobian[k * MATRIX_SIZE + i];
        jacobian[k * MATRIX_SIZE + j] -= jacobian[i * MATRIX_SIZE + j] * factor;
        inverse[k * MATRIX_SIZE + j] -= inverse[i * MATRIX_SIZE + j] * factor;
    }
}

__global__ void initIdentity(double* inverse, int MATRIX_SIZE) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < MATRIX_SIZE) {
        for (int j = 0; j < MATRIX_SIZE; ++j)
            inverse[i * MATRIX_SIZE + j] = (i == j) ? 1.0 : 0.0;
    }
}

void gpu_inverse(double* jacobian_d, double* inverse_jacobian_d, int MATRIX_SIZE) {
    int x_blocks_count = (MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, 1, 1);
    initIdentity << <gridDim, blockDim >> > (inverse_jacobian_d, MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        double pivot;
        cudaMemcpy(&pivot, &jacobian_d[i * MATRIX_SIZE +i], sizeof(double), cudaMemcpyDeviceToHost);
        normalizeRow << <gridDim, blockDim >> > (jacobian_d, inverse_jacobian_d, i, pivot, MATRIX_SIZE);
        eliminateColumn << <MATRIX_SIZE, MATRIX_SIZE >> > (jacobian_d, inverse_jacobian_d, i, MATRIX_SIZE);
    }
}

void NewtonSolver::gpu_newton_solve() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int version = prop.major;
    FileOperations* file_op = new FileOperations();
	std::string file_name = "gpu_newton_solver_" + std::to_string(data->MATRIX_SIZE) + ".csv";
	file_op->create_file(file_name, 5);
    file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");

    NewtonSolverFunctions::gpu_dummy_warmup << <1, 32 >> > ();
    cudaDeviceSynchronize();
    std::cout << "GPU Newton solver\n";
    int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iterations_count = 0;
    double dx = 0;

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

    double* delta = new double[data->MATRIX_SIZE];

    auto start_total = std::chrono::high_resolution_clock::now();

    cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t myStream;
    cudaStreamCreate(&myStream);

    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
        auto start = std::chrono::high_resolution_clock::now();
#endif
		std::cout << "Power: " << data->equation->get_power() << "\n";
        NewtonSolverFunctions::gpu_compute_func_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
            data->points_d, data->indexes_d, data->intermediate_funcs_value_d, data->MATRIX_SIZE, version, data->equation->get_power());
        cudaDeviceSynchronize();

        cudaMemcpy(data->intermediate_funcs_value_h, data->intermediate_funcs_value_d, x_blocks_count * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < data->MATRIX_SIZE; i++) {
            data->funcs_value_h[i] = -data->vector_b_h[i];
            for (int j = 0; j < x_blocks_count; j++) {
                data->funcs_value_h[i] += data->intermediate_funcs_value_h[i * x_blocks_count + j];
            }
        }
        cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
        auto end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        NewtonSolverFunctions::gpu_compute_jacobian << <gridDim, blockDim, 2 * blockDim.x * sizeof(double) >> > (
            data->points_d, data->indexes_d, data->jacobian_d, data->MATRIX_SIZE, data->equation->get_power());
        cudaDeviceSynchronize();

        cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif
        gpu_cublasInverse(data, myStream);
        cudaDeviceSynchronize();
		//gpu_inverse(data->jacobian_d, data->inverse_jacobian_d);
#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        //cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
        NewtonSolverFunctions::gpu_compute_delta_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
            data->funcs_value_d, data->inverse_jacobian_d, data->delta_d, data->MATRIX_SIZE, version);
        cudaDeviceSynchronize();

        cudaMemcpy(data->delta_h, data->delta_d, x_blocks_count * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

        for (int i = 0; i < data->MATRIX_SIZE; i++) {
            delta[i] = 0;
            for (int j = 0; j < x_blocks_count; j++) {
                delta[i] -= data->delta_h[i * x_blocks_count + j];
            }
        }

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
        start = std::chrono::high_resolution_clock::now();
#endif

        dx = 0.0;
        for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] += delta[i];
            dx = std::max(dx, std::abs(delta[i]));
        }

        cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
        end = std::chrono::high_resolution_clock::now();
        data->intermediate_results[4] = std::chrono::duration<double>(end - start).count();

        std::cout << "\nIteration: " << iterations_count << "\n";
        std::cout << "===============================================================\n";
        std::cout << "Intermediate results: \n";
        std::cout << "Compute func values: " << data->intermediate_results[0] << "s" << "\n";
        std::cout << "Compute jacobian: " << data->intermediate_results[1] << "s" << "\n";
        std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "s" << "\n";
        std::cout << "Compute delta: " << data->intermediate_results[3] << "s" << "\n";
        std::cout << "Update points: " << data->intermediate_results[4] << "s" << "\n";
		std::cout << "Error (dx): " << dx << "\n";
        std::cout << "===============================================================\n";
#endif
        file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
    } while (dx > TOLERANCE);
	file_op->close_file();

    auto end_total = std::chrono::high_resolution_clock::now();
    data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();


    print_solution(iterations_count, data->points_h, data->points_check);
    cudaStreamDestroy(myStream);
    delete[] delta;
}