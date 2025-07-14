﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "math.h"
#include "vector" 
#include "DataInitializerCPU.h"
#include "memory"
#include "config.h"
#include "NewtonSolverCuDSS.h"
#include "FileOperations.h"
#include <cstdlib>
#include "NewtonSolverCUDA.h"
#include <DataInitializerCuDSS.h>
#include <NewtonSolverCPU.h>

int main(int argc, char* argv[]) {

    int matrix_size_max = 1000;
    int matrix_size_min = 1000;
    int stride = 100;
    int power = 3;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];

        if (arg.find("--power=") == 0) {
            power = std::stod(arg.substr(8));
        }
    }
    if (argc > 1) {
        matrix_size_max = std::atoi(argv[1]);

        if (argc > 2) {
            matrix_size_min = std::atoi(argv[2]);

            if (argc > 3) {
                stride = std::atoi(argv[3]);
            }
        }
    }

    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>();
    std::string header = "CPU,GPU,cuDSS,matrix_size";
    file_op->create_file("total_statistic.csv", 3);
    file_op->append_file_headers(header);
    std::vector<double> row(3);

    for (int size = matrix_size_min; size <= matrix_size_max; size += stride) {

        //
        // CPY
        //
        {
            std::unique_ptr<DataInitializerCPU> data = std::make_unique<DataInitializerCPU>(size, 0, size, power);
            std::unique_ptr<NewtonSolverCPU> newton_solver = std::make_unique<NewtonSolverCPU>(data.get());
            newton_solver->cpu_newton_solve();
            row[0] = 0;
        }

        //
        // GPU
        //
        {
            std::unique_ptr<DataInitializerCUDA> data2 = std::make_unique<DataInitializerCUDA>(size, 0, size, power);
            std::unique_ptr<NewtonSolverCUDA> newton_solver2 = std::make_unique<NewtonSolverCUDA>(data2.get());
            newton_solver2->gpu_newton_solve();
            row[1] = data2->total_elapsed_time;;
        }

        //
        // cuDSS
        //
        {
            std::unique_ptr<DataInitializerCuDSS> data3 = std::make_unique<DataInitializerCuDSS>(size, 0, size, power);
            std::unique_ptr<NewtonSolverCuDSS> cuDssSolver = std::make_unique<NewtonSolverCuDSS>(data3.get());
            cuDssSolver->gpu_newton_solver_cudss();
            row[2] = data3->total_elapsed_time;;
        }
        file_op->append_file_data(row, size);
    }
    return 0;
}
