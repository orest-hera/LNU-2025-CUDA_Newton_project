#include "EditionalTools.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random> 
#include <vector>

#include "DataInitializer.h"

template<class It, class G>
void random_shuffle(It first, It last, G& gen)
{
    for (auto i = last - first - 1; i > 0; --i)
    {
        std::swap(first[i], first[gen() % (i + 1)]);
    }
}

static std::vector<double> generate_row(
        std::mt19937& gen, std::vector<int>& pos, double &b, int row, int size,
        int nnz, const double *points, const Equation* eqn)
{
    auto rand_max = gen.max();

    std::vector<int> p;
    p.reserve(size);

    p.push_back(row);

    for (int i = 0; i < size; ++i) {
        if (i != row) {
            p.push_back(i);
        }
    }

    random_shuffle(p.begin() + 1, p.end(), gen);

    pos.clear();
    std::copy(p.begin(), p.begin() + nnz, std::back_inserter(pos));
    std::sort(pos.begin(), pos.end());

    std::vector<double> vals;
    vals.reserve(nnz);

    for (int i = 0; i < nnz; ++i) {
        double v = static_cast<double>(gen()) / rand_max;
        vals.push_back(v);
    }

    double sum = 0.0;
    int diagonal_idx = -1;

    for (int idx = 0; idx < nnz; idx++) {
        if (pos[idx] == row) {
            diagonal_idx = idx;
        }
        else {
            sum += std::abs(vals[idx]);
        }
    }

    if (diagonal_idx != -1) {
        if ((sum + 1e-7) >= std::abs(vals[diagonal_idx])) {
            vals[diagonal_idx] = sum + 1.0;
        }
    }

    b = 0;
    for (int i = 0; i < nnz; ++i) {
        b += eqn->calculate_term_value(vals[i], points[pos[i]]);
    }

    return vals;
}


void tools::generate_sparse_initial_indexes_matrix_and_vector_b(
    double* matrix,
    double* b,
    double* points,
    int MATRIX_SIZE,
    Equation* equation,
    int zero_elements_per_row)
{
    if (zero_elements_per_row < 0 || zero_elements_per_row >= MATRIX_SIZE) {
        zero_elements_per_row = 0;
    }

    std::mt19937 gen;
    auto rand_max = gen.max();

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points[i] = static_cast<double>(gen()) / rand_max;
    }

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::vector<int> selected_positions;
        double bb;

        int non_zero_count = MATRIX_SIZE - zero_elements_per_row;

        std::vector<double> values = generate_row(
                    gen, selected_positions, bb, i, MATRIX_SIZE, non_zero_count,
                    points, equation);

        size_t idx = 0;

        for (int col : selected_positions) {
            matrix[i * MATRIX_SIZE + col] = values[idx];
            idx++;
        }

        b[i] = bb;
    }
}

void tools::generate_sparse_initial_indexes_matrix_and_vector_b(
    double* csr_values,
    int* csr_rows,
    int* csr_cols,
    double* b,
    double* points,
    int MATRIX_SIZE,
    Equation* equation,
    int zero_elements_per_row)
{
    if (zero_elements_per_row < 0 || zero_elements_per_row >= MATRIX_SIZE) {
        zero_elements_per_row = 0;
    }

    std::mt19937 gen;
    auto rand_max = gen.max();

    for (int i = 0; i < MATRIX_SIZE; i++) {
        points[i] = static_cast<double>(gen()) / rand_max;
    }

    csr_rows[0] = 0;
    int current_pos = 0;

    for (int i = 0; i < MATRIX_SIZE; i++) {
        std::vector<int> selected_positions;
        double bb;

        int non_zero_count = MATRIX_SIZE - zero_elements_per_row;

        std::vector<double> values = generate_row(
                    gen, selected_positions, bb, i, MATRIX_SIZE, non_zero_count,
                    points, equation);

        size_t idx = 0;

        for (int col : selected_positions) {
            csr_cols[current_pos] = col;
            csr_values[current_pos] = values[idx];
            current_pos++;
            idx++;
        }

        csr_rows[i + 1] = current_pos;
        b[i] = bb;
    }
}



double tools::calculate_index_xn(double index, double x) {
	return index * x;
}

void tools::print_solution(DataInitializer* data, int iterations_count = 0) {
    std::cout << "Total Iterations count: " << iterations_count << "\n";

#ifdef TOTAL_ELASPED_TIME
    std::cout << "Total elapsed time: " << data->total_elapsed_time << "\n";
#endif
#ifdef SOLUTION_PRINT
    std::cout << "Solution: \n";

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        std::cout << data->points_h[i] << " " << data->points_check[i] << "\n";
    }
#endif
    std::cout << "\n";

    bool isCorrect = true;

    for (int i = 0; i < data->MATRIX_SIZE; i++) {
        if (std::abs(data->points_h[i] - data->points_check[i]) > 1e-5) {
            isCorrect = false;
        }
    }

    if (isCorrect) {
        std::cout << "Solution is correct!" << "\n";
    }
    else {
        std::cout << "Solution is incorrect!" << "\n";
    }
}

void tools::print_intermediate_result(DataInitializer* data, int iteration_number, double error, bool isCudss) {
    std::cout << "\nIteration: " << iteration_number << "\n";
    std::cout << "===============================================================\n";
    std::cout << "Intermediate results: \n";
    std::cout << "Compute func values: " << data->intermediate_results[0] << "\n";
    std::cout << "Compute jacobian: " << data->intermediate_results[1] << "\n";
    std::cout << "Compute inverse jacobian: " << data->intermediate_results[2] << "\n";
    std::cout << "Compute delta: " << data->intermediate_results[3] << "\n";
    if (!isCudss) {
        std::cout << "Update points: " << data->intermediate_results[4] << "\n";
    }
    std::cout << "Error: " << error << "\n";
    std::cout << "===============================================================\n";
}
