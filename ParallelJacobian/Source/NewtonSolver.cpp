#include "math.h"
#include <iostream>
#include <algorithm>
#include <cusolverDn.h>
#include "FileOperations.h"
#include "NewtonSolver.h"
#include "EditionalTools.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

NewtonSolver::NewtonSolver(DataInitializer* dataInitializer) {
	data = dataInitializer;
}

NewtonSolver::~NewtonSolver() {
}

void NewtonSolver::cpu_computeVec() {
	for (int i = 0; i < data->MATRIX_SIZE; i++) {
        data->funcs_value_h[i] = -data->vector_b_h[i];
		for (int j = 0; j < data->MATRIX_SIZE; j++) {
			data->funcs_value_h[i] += tools::calculate_index_xn(data->indexes_h[i * data->MATRIX_SIZE + j], data->points_h[j]);
        }
	}
}

double NewtonSolver::cpu_compute_derivative(int rowIndex, int colIndex) {
    double* temp_plus, *temp_minus;
	temp_plus = new double[data->MATRIX_SIZE];
	temp_minus = new double[data->MATRIX_SIZE];

    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        temp_plus[i] = data->points_h[i];
        temp_minus[i] = data->points_h[i];
    }

    double equrency = EQURENCY;

    temp_plus[colIndex] += equrency;
    temp_minus[colIndex] -= equrency;

    double f_plus = 0.0, f_minus = 0.0;
    for (int j = 0; j < data->MATRIX_SIZE; ++j) {
        f_plus += tools::calculate_index_xn(data->indexes_h[rowIndex * data->MATRIX_SIZE + j], temp_plus[j]);
        f_minus += tools::calculate_index_xn(data->indexes_h[rowIndex * data->MATRIX_SIZE + j], temp_minus[j]);
    }

    return (f_plus - f_minus) / (2.0 * equrency);
}

//
//   CPU
//

void NewtonSolver::cpu_compute_jacobian() {
    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        for (int j = 0; j < data->MATRIX_SIZE; ++j) {
            data->jacobian_h[i * data->MATRIX_SIZE + j] = cpu_compute_derivative(i, j);
        }
    }
}

void NewtonSolver::cpu_inverse() {
    for (int i = 0; i < data->MATRIX_SIZE; i++) data->inverse_jacobian_h[i * data->MATRIX_SIZE + i] = 1.0;

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        double temp = data->jacobian_h[i * data->MATRIX_SIZE + i];
        for (int j = 0; j < data->MATRIX_SIZE; j++) {
            data->jacobian_h[i * data->MATRIX_SIZE + j] /= temp;
            data->inverse_jacobian_h[i * data->MATRIX_SIZE + j] /= temp;
        }
        for (int k = 0; k < data->MATRIX_SIZE; k++) {
            if (k != i) {
                temp = data->jacobian_h[k * data->MATRIX_SIZE + i];
                for (int j = 0; j < data->MATRIX_SIZE; j++) {
                    data->jacobian_h[k * data->MATRIX_SIZE + j] -= data->jacobian_h[i * data->MATRIX_SIZE + j] * temp;
                    data->inverse_jacobian_h[k * data->MATRIX_SIZE + j] -= data->inverse_jacobian_h[i * data->MATRIX_SIZE + j] * temp;
                }
            }
        }
    }
}

void NewtonSolver::cpu_compute_delta() {
    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        data->delta_h[i] = 0.0;
        for (int j = 0; j < data->MATRIX_SIZE; j++) {
            data->delta_h[i] -= tools::calculate_index_xn(data->inverse_jacobian_h[i * data->MATRIX_SIZE + j], data->funcs_value_h[j]);
        }
    }
}

void NewtonSolver::cpu_newton_solve() {
    std::cout << "CPU Newton solver\n";
    double dx = 0.0;
    int iterations_count = 0;
    FileOperations* file_op = new FileOperations();
    std::string file_name = "cpu_newton_solver_" + std::to_string(data->MATRIX_SIZE) + ".csv";
    file_op->create_file(file_name, 5);
    file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");


#ifdef TOTAL_ELASPED_TIME
	auto start_total = std::chrono::high_resolution_clock::now();
#endif
    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::high_resolution_clock::now();
#endif
        cpu_computeVec();
#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
        data->intermediate_results[0] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_compute_jacobian();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[1] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_inverse();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[2] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::high_resolution_clock::now();
#endif
        cpu_compute_delta();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
        data->intermediate_results[3] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
        start = std::chrono::high_resolution_clock::now();
#endif
        dx = 0;
        for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] += data->delta_h[i];
            //data->points_h[i] = tools::calculate_index_xn(data->indexes_h[i * data->MATRIX_SIZE + i], data->points_h[i]);
            dx = std::max(dx, std::abs(data->delta_h[i]));
        }
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::high_resolution_clock::now();
		elapsed = end - start;
		data->intermediate_results[4] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		std::cout << "\nIteration: " << iterations_count << "\n";
        std::cout << "===============================================================\n";
        std::cout << "Intermediate results: \n";
        std::cout << "Compute func values: " << data->intermediate_results[0] << "\n";
        std::cout << "Compute jacobian: " << data->intermediate_results[1] << "\n";
        std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "\n";
        std::cout << "Compute delta: " << data->intermediate_results[3] << "\n";
        std::cout << "Update points: " << data->intermediate_results[4] << "\n";
		std::cout << "Error: " << dx << "\n";
        std::cout << "===============================================================\n";
#endif
        file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
    } while (dx > TOLERANCE);
    file_op->close_file();

	auto end_total = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_total = end_total - start_total;
	data->total_elapsed_time = elapsed_total.count();

    print_solution(iterations_count, data->points_h);
}

//
// GPU
//

#ifdef GPU_SOLVER
//void gpu_cusolverInverse(DataInitializer* data) {
//    cusolverDnHandle_t cusolverH;
//    cusolverDnCreate(&cusolverH);
//
//    int work_size = 0;
//    int* dev_info;
//    cudaMalloc((void**)&dev_info, sizeof(int));
//
//    cusolverDnDgetrf_bufferSize(
//        cusolverH,
//        MATRIX_SIZE,
//        MATRIX_SIZE,
//        data->jacobian_d,
//        MATRIX_SIZE,
//        &work_size
//    );
//
//    double* d_work;
//    cudaMalloc((void**)&d_work, work_size * sizeof(double));
//
//    double* d_LU;
//    cudaMalloc((void**)&d_LU, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
//    cudaMemcpy(d_LU, data->jacobian_d, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToDevice);
//
//    int* d_pivot;
//    cudaMalloc((void**)&d_pivot, MATRIX_SIZE * sizeof(int));
//
//    cusolverDnDgetrf(
//        cusolverH,
//        MATRIX_SIZE,
//        MATRIX_SIZE,
//        d_LU,
//        MATRIX_SIZE,
//        d_work,
//        d_pivot,
//        dev_info
//    );
//
//    cudaMemset(data->inverse_jacobian_d, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
//    double temp = 1;
//    for (int i = 0; i < MATRIX_SIZE; i++) {
//        cudaMemcpy(data->inverse_jacobian_d + i * MATRIX_SIZE + i, &temp, sizeof(double), cudaMemcpyHostToDevice);
//    }
//
//    cusolverDnDgetrs(
//        cusolverH,
//        CUBLAS_OP_N,
//        MATRIX_SIZE,
//        MATRIX_SIZE,
//        d_LU,
//        MATRIX_SIZE,
//        d_pivot,
//        data->inverse_jacobian_d,
//        MATRIX_SIZE,
//        dev_info
//    );
//
//    cudaFree(d_work);
//    cudaFree(d_pivot);
//    cudaFree(d_LU);
//    cudaFree(dev_info);
//
//    cusolverDnDestroy(cusolverH);
//}

void gpu_cublasInverse(DataInitializer* data, cudaStream_t stream) {
    cublasSetStream(data->cublasContextHandler, stream);
    cublasStatus_t status2 = cublasDgetrfBatched(data->cublasContextHandler, data->MATRIX_SIZE, data->cublas_ajacobian_d, data->MATRIX_SIZE, data->cublas_pivot, data->cublas_info, 1);
    
    cudaStreamSynchronize(stream);

    cublasStatus_t status = cublasDgetriBatched(data->cublasContextHandler, data->MATRIX_SIZE, (const double**)data->cublas_ajacobian_d, data->MATRIX_SIZE, data->cublas_pivot, data->cublas_ainverse_jacobian_d, data->MATRIX_SIZE, data->cublas_info, 1);
    cudaStreamSynchronize(stream);
}
#endif

void NewtonSolver::print_solution(int iterations_count, double* result) {
    std::cout << "Total Iterations count: " << iterations_count << "\n";

#ifdef TOTAL_ELASPED_TIME
	std::cout << "Total elapsed time: " << data->total_elapsed_time << "\n";
#endif
#ifdef SOLUTION_PRINT
    std::cout << "Solution: \n";

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        std::cout << result[i] << "\n";
    }
#endif
    std::cout << "\n";
}