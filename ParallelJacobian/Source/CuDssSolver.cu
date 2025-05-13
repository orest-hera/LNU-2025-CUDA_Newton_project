#include "CuDssSolver.h"
#include "NewtonSolverFunctions.h"
#include "DataInitializer.h"
#include "FileOperations.h"
#include "iostream"
#include "math.h"
#include "chrono"


void CuDssSolver::gpu_newton_solver_cudss() {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int version = prop.major;
	FileOperations* file_op = new FileOperations();
	std::string file_name = "gpu_cudss_newton_solver_" + std::to_string(data->MATRIX_SIZE) + ".csv";
	file_op->create_file(file_name, 5);
	file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");

	NewtonSolverFunctions::gpu_dummy_warmup << <1, 32 >> > ();
	cudaDeviceSynchronize();
	std::cout << "GPU CuDss Newton solver\n";
	int x_blocks_count = (data->MATRIX_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
	int iterations_count = 0;
	double dx = 0;

	dim3 blockDim(BLOCK_SIZE, 1, 1);
	dim3 gridDim(x_blocks_count, data->MATRIX_SIZE, 1);

	auto start_total = std::chrono::high_resolution_clock::now();

	cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(data->indexes_d, data->indexes_h, data->MATRIX_SIZE * data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
		data->points_h[i] += data->delta_h[i];
		dx = std::max(dx, std::abs(data->delta_h[i]));
	}

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
		for (int i = 0; i < data->MATRIX_SIZE; i++) {
			data->funcs_value_h[i] *= -1;
		}
		cudaMemcpy(data->funcs_value_d, data->funcs_value_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);

		solve(data->jacobian_h, data->funcs_value_d, data->delta_h, data->delta_d);
		dx = 0.0;
		for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] += data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}
		std::cout << "dx: " << dx << "\n";
		cudaMemcpy(data->points_d, data->points_h, data->MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice);
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::high_resolution_clock::now();
	auto total_time = std::chrono::duration<double>(end_total - start_total).count();
	std::cout << "Total time: " << total_time << "\n";
	print_solutionn(iterations_count, data->points_h, data->points_check, data);
}