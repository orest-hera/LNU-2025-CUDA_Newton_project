#include "stdio.h"
#include <iostream>
#include <memory>
#include <string>
#include "cuda_runtime.h"
#include "FileOperations.h"
#include "NewtonSolverGPUFunctions.h"
#include <EditionalTools.h>
#include "NewtonSolverCUDA.h"
#include <chrono>

NewtonSolverCUDA::NewtonSolverCUDA(DataInitializerCUDA* dataInitializer) {
	data = dataInitializer;
}

NewtonSolverCUDA::~NewtonSolverCUDA() {
}
void NewtonSolverCUDA::gpu_cublasInverse(DataInitializerCUDA* data) {
    cublasStatus_t status2 = cublasDgetrfBatched(data->cublasContextHandler, data->MATRIX_SIZE, data->cublas_ajacobian_d, data->MATRIX_SIZE, data->cublas_pivot, data->cublas_info, 1);
    cublasStatus_t status = cublasDgetriBatched(data->cublasContextHandler, data->MATRIX_SIZE, (const double**)data->cublas_ajacobian_d, data->MATRIX_SIZE, data->cublas_pivot, data->cublas_ainverse_jacobian_d, data->MATRIX_SIZE, data->cublas_info, 1);
}

void NewtonSolverCUDA::gpu_newton_solve() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int version = prop.major;
    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>();
    std::string file_name = "gpu_newton_solver_" + std::to_string(data->file_name) + ".csv";
    file_op->create_file(file_name, 5);
    file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");

    NewtonSolverGPUFunctions::gpu_dummy_warmup << <1, 32 >> > ();
    cudaDeviceSynchronize();
    std::cout << "GPU Newton solver\n";
    int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int iterations_count = 0;
    double dx = 0;

    dim3 blockDim(BLOCK_SIZE, 1, 1);
    dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

    double* delta = new double[data->MATRIX_SIZE];

	auto start_total = std::chrono::steady_clock::now();

    cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    cudaStream_t myStream;
    cudaStreamCreate(&myStream);

    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif
		std::cout << "Power: " << data->equation->get_power() << "\n";
        NewtonSolverGPUFunctions::gpu_compute_func_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
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
		auto end = std::chrono::steady_clock::now();
        data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

        NewtonSolverGPUFunctions::gpu_compute_jacobian << <gridDim, blockDim >> > (
            data->points_d, data->indexes_d, data->jacobian_d, data->MATRIX_SIZE, data->equation->get_power());
        cudaDeviceSynchronize();

        //cudaMemcpy(data->jacobian_h, data->jacobian_d, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
        data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
        gpu_cublasInverse(data);
        cudaDeviceSynchronize();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
        data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

        NewtonSolverGPUFunctions::gpu_compute_delta_values << <gridDim, blockDim, blockDim.x * sizeof(double) >> > (
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
		end = std::chrono::steady_clock::now();
        data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

        dx = 0.0;
        for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] += delta[i];
            dx = std::max(dx, std::abs(delta[i]));
        }

        cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
        data->intermediate_results[4] = std::chrono::duration<double>(end - start).count();

		tools::print_intermediate_result(data, iterations_count, dx);
#endif
        file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
    } while (dx > TOLERANCE);
	file_op->close_file();

	auto end_total = std::chrono::steady_clock::now();
    data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();

    tools::print_solution(data, iterations_count);
    cudaStreamDestroy(myStream);
    delete[] delta;
}
