#include "math.h"
#include <iostream>
#include <memory>

#include "FileOperations.h"
#include "NewtonSolverCPU.h"
#include "EditionalTools.h"

NewtonSolverCPU::NewtonSolverCPU(DataInitializerCPU* dataInitializer) {
	data = dataInitializer;
}

NewtonSolverCPU::~NewtonSolverCPU() {
}

void NewtonSolverCPU::cpu_computeVec() {
    const int N = data->MATRIX_SIZE;

    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int j = 0; j < N; ++j) {
            double index = data->indexes_h[i * N + j];
            double x = data->points_h[j];
            sum += data->equation->calculate_term_value(index, x);
        }

        data->funcs_value_h[i] = sum - data->vector_b_h[i];
    }
}

double NewtonSolverCPU::cpu_compute_derivative(int rowIndex, int colIndex) {
    double value = data->points_h[colIndex];
    double element = data->indexes_h[rowIndex * data->MATRIX_SIZE + colIndex];

    double f_plus =  data->equation->calculate_term_value(element, value + EQURENCY);
    double f_minus = data->equation->calculate_term_value(element, value - EQURENCY);;

    return (f_plus - f_minus) / (2.0 * EQURENCY);
}

void NewtonSolverCPU::cpu_compute_jacobian() {
    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        for (int j = 0; j < data->MATRIX_SIZE; ++j) {
            data->jacobian_h[i * data->MATRIX_SIZE + j] = cpu_compute_derivative(i, j);
        }
    }
}

void NewtonSolverCPU::cpu_inverse() {
    int N = data->MATRIX_SIZE;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            data->inverse_jacobian_h[i * N + j] = (i == j) ? 1.0 : 0.0;

    for (int i = 0; i < N; i++) {
        int maxRow = i;
        double maxVal = fabs(data->jacobian_h[i * N + i]);
        for (int k = i + 1; k < N; k++) {
            double val = fabs(data->jacobian_h[k * N + i]);
            if (val > maxVal) {
                maxVal = val;
                maxRow = k;
            }
        }

        if (maxRow != i) {
            for (int j = 0; j < N; j++) {
                std::swap(data->jacobian_h[i * N + j], data->jacobian_h[maxRow * N + j]);
                std::swap(data->inverse_jacobian_h[i * N + j], data->inverse_jacobian_h[maxRow * N + j]);
            }
        }

        double temp = data->jacobian_h[i * N + i];
        if (fabs(temp) < 1e-12) {
            std::cerr << "Jacobian is singular or nearly singular at row " << i << std::endl;
            return;
        }

        for (int j = 0; j < N; j++) {
            data->jacobian_h[i * N + j] /= temp;
            data->inverse_jacobian_h[i * N + j] /= temp;
        }

        for (int k = 0; k < N; k++) {
            if (k != i) {
                double factor = data->jacobian_h[k * N + i];
                for (int j = 0; j < N; j++) {
                    data->jacobian_h[k * N + j] -= data->jacobian_h[i * N + j] * factor;
                    data->inverse_jacobian_h[k * N + j] -= data->inverse_jacobian_h[i * N + j] * factor;
                }
            }
        }
    }
}


void NewtonSolverCPU::cpu_compute_delta() {
    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        data->delta_h[i] = 0.0;
        for (int j = 0; j < data->MATRIX_SIZE; j++) {
            data->delta_h[i] -= data->inverse_jacobian_h[i * data->MATRIX_SIZE + j] * data->funcs_value_h[j];
        }
    }
}

void NewtonSolverCPU::cpu_newton_solve() {
    std::cout << "CPU Newton solver\n";
    double dx = 0.0;
    int iterations_count = 0;
    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>();
    std::string file_name = "cpu_newton_solver_" + std::to_string(data->file_name) + ".csv";
    file_op->create_file(file_name, 5);
    file_op->append_file_headers("func_value_t,jacobian_value_t,inverse_jacobian_t,delta_value_t,update_points_t,matrix_size");


#ifdef TOTAL_ELASPED_TIME
	auto start_total = std::chrono::steady_clock::now();
#endif
    do {
        iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif
        cpu_computeVec();
#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::steady_clock::now();
		std::chrono::duration<double> elapsed = end - start;
        data->intermediate_results[0] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::steady_clock::now();
#endif
        cpu_compute_jacobian();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
        data->intermediate_results[1] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::steady_clock::now();
#endif
        cpu_inverse();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
        data->intermediate_results[2] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::steady_clock::now();
#endif
        cpu_compute_delta();
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
        data->intermediate_results[3] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		start = std::chrono::steady_clock::now();
#endif
        dx = 0;
        for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
            data->points_h[i] += data->delta_h[i];
            dx = std::max(dx, std::abs(data->delta_h[i]));
        }
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		elapsed = end - start;
		data->intermediate_results[4] = elapsed.count();
#endif

#ifdef INTERMEDIATE_RESULTS
		tools::print_intermediate_result(data, iterations_count, dx);
#endif

        file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
    } while (dx > TOLERANCE);
    file_op->close_file();

	auto end_total = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_total = end_total - start_total;
	data->total_elapsed_time = elapsed_total.count();

    tools::print_solution(data, iterations_count);
}
