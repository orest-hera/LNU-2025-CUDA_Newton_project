#include "NewtonSolverMKLdss.h"
#include "FileOperations.h"
#include "iostream"
#include "chrono"
#include <memory>
#include "EditionalTools.h"
#include "mkl.h"


int NewtonSolverMKLdss::count_non_zero_elements(double* matrix_A) {
	int non_zero_count = 0;
	for (int i = 0; i < data->MATRIX_SIZE * data->MATRIX_SIZE; i++) {
		if (matrix_A[i] != 0) {
			non_zero_count++;
		}
	}
	return non_zero_count;
}

void NewtonSolverMKLdss::parse_to_csr(int* csr_cols, int* csr_rows, double* csr_values, double* matrix_A) {
	int non_zero_count = 0;
	csr_rows[0] = 0;
	for (int i = 0; i < data->MATRIX_SIZE; ++i) {
		for (int j = 0; j < data->MATRIX_SIZE; ++j) {
			if (matrix_A[i * data->MATRIX_SIZE + j] != 0) {
				csr_cols[non_zero_count] = j;
				csr_values[non_zero_count] = matrix_A[i * data->MATRIX_SIZE + j];
				non_zero_count++;
			}
		}
		csr_rows[i + 1] = non_zero_count;
	}
}

void NewtonSolverMKLdss::cpu_computeVec() {
    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        double sum = 0.0;
        int row_start = data->csr_rows_h[i];
        int row_end = data->csr_rows_h[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int col = data->csr_cols_h[idx];
            double index = data->csr_values_h[idx];
            double x = data->points_h[col];

            sum += data->equation->calculate_term_value(index, x);
        }

        data->funcs_value_h[i] = -(sum - data->vector_b_h[i]);
    }
}

double NewtonSolverMKLdss::cpu_compute_derivative(int elementIndex, int colIndex) {
    double value = data->points_h[colIndex];
    double element = data->csr_values_h[elementIndex];

    double f_plus =  data->equation->calculate_term_value(element, value + EQURENCY);
    double f_minus = data->equation->calculate_term_value(element, value - EQURENCY);

    return (f_plus - f_minus) / (2.0 * EQURENCY);
}

void NewtonSolverMKLdss::cpu_compute_jacobian() {

    for (int i = 0; i < data->MATRIX_SIZE; ++i) {
        int row_start = data->csr_rows_h[i];
        int row_end = data->csr_rows_h[i + 1];

        for (int idx = row_start; idx < row_end; ++idx) {
            int j = data->csr_cols_h[idx];
            data->jacobian[idx] = cpu_compute_derivative(idx, j);
        }
    }
}

void NewtonSolverMKLdss::cpu_mkl_dss_find_delta() {
	int info;
	if (!data->analyzed){
		static const MKL_INT size = data->MATRIX_SIZE;
		static const MKL_INT nnz = data->csr_rows_h[data->MATRIX_SIZE];
		// for (int i = 0; i < data->MATRIX_SIZE; i++){
		// 	data->csr_rows_h[i] += 1;
		// 	for (int j = 0; j < data->MATRIX_SIZE; j++){
		// 		data->csr_cols_h[i * data->MATRIX_SIZE + j] += 1;
		// 	}
		// }
		// static MKL_INT* csr_cols = new MKL_INT[data->csr_rows_h[data->MATRIX_SIZE]];
		// static MKL_INT* csr_rows = new MKL_INT[data->MATRIX_SIZE + 1];
		info = dss_define_structure(data->handle, data->sym, data->csr_rows_h, 
							size, size, data->csr_cols_h,
		nnz);
		std::cout << "MATRIX_SIZE" << data->MATRIX_SIZE << std::endl;
		std::cout << data->non_zero_count << " " << data->csr_rows_h[data->MATRIX_SIZE] << std::endl;
		
		
						// int perm = 0;
		static MKL_INT* perm = new MKL_INT[data->MATRIX_SIZE]();
		// for (int i = 0; i < data->MATRIX_SIZE; i++){
		// 	perm[i] = 0;
		// }
		std::cout << "dss_define_structure : " << info << std::endl;		
		info = dss_reorder(data->handle, data->order, perm);
						data->analyzed = true;
						std::cout << "dss_reorder : " << info << std::endl;
	}

	info = dss_factor_real(data->handle, data->type, data->jacobian);
	std::cout << "dss_factor_real : " << info << std::endl;

	info = dss_solve_real(data->handle, data->opt, data->funcs_value_h, data->MATRIX_SIZE, data->delta_h);
	std::cout << "dss_solve_real : " << info << std::endl;
}

void NewtonSolverMKLdss::cpu_newton_solve() {
	std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(settings_.path);
	std::string file_name = "cpu_mkl_dss_newton_solver_" + std::to_string(data->file_name) + ".csv";
	file_op->create_file(file_name, 4);
	file_op->append_file_headers("func_value_t,jacobian_value_t,delta_value_t,update_points_t,matrix_size");

	std::cout << "CPU MKL dss Newton solver\n";
	std::cout << "Power: " << data->equation->get_power() << "\n";
	int iterations_count = 0;
	double dx = 0;

	auto start_total = std::chrono::steady_clock::now();

	
	parse_to_csr(data->csr_cols_h, data->csr_rows_h, data->csr_values_h, data->indexes_h);
	std::cout << "Non-zero count: " << data->non_zero_count << "\n";
	do {
		iterations_count++;

#ifdef INTERMEDIATE_RESULTS
		auto start = std::chrono::steady_clock::now();
#endif

		cpu_computeVec();	
		std::cout << "Vec" << std::endl;

#ifdef INTERMEDIATE_RESULTS
		auto end = std::chrono::steady_clock::now();
		data->intermediate_results[0] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		cpu_compute_jacobian();
		std::cout << "Jacobian" << std::endl;

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[1] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif

		cpu_mkl_dss_find_delta();
		std::cout << "Delta" << std::endl;
#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[2] = std::chrono::duration<double>(end - start).count();
		start = std::chrono::steady_clock::now();
#endif
		dx = 0.0;
		for (size_t i = 0; i < data->MATRIX_SIZE; ++i) {
			data->points_h[i] -= data->delta_h[i];
			dx = std::max(dx, std::abs(data->delta_h[i]));
		}

		for (int i = 0; i < data->MATRIX_SIZE; i++) {
			std::cout << data->points_h[i] << std::endl;
		}

#ifdef INTERMEDIATE_RESULTS
		end = std::chrono::steady_clock::now();
		data->intermediate_results[3] = std::chrono::duration<double>(end - start).count();
#endif
		tools::print_intermediate_result(data, iterations_count, dx, true);
		file_op->append_file_data(data->intermediate_results, data->MATRIX_SIZE);
		//if (iterations_count == 4) {
		//	break;
		//}
	} while (dx > TOLERANCE);

	auto end_total = std::chrono::steady_clock::now();
	data->total_elapsed_time = std::chrono::duration<double>(end_total - start_total).count();
	tools::print_solution(data, iterations_count);
}

NewtonSolverMKLdss::NewtonSolverMKLdss(DataInitializerMKLdss* data,
		const Settings::SettingsData& settings)
	: settings_{settings}
{
	this->data = data;
}

NewtonSolverMKLdss::~NewtonSolverMKLdss() {
}
