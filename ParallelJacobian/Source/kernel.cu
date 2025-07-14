#include <iostream>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "DataInitializerCPU.h"
#include "DataInitializerCuDSS.h"
#include "NewtonSolverCPU.h"
#include "NewtonSolverCUDA.h"
#include "NewtonSolverCuDSS.h"
#include "FileOperations.h"
#include "config.h"
#include "settings.h"

int main(int argc, char* argv[]) {

    Settings s;
    if (!s.parse(argc, argv))
        return 1;

    int matrix_size_max = s.settings.max;
    int matrix_size_min = s.settings.min;
    int stride = s.settings.stride;
    int power = s.settings.power;

    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>();
    std::string header = "CPU,GPU,cuDSS,matrix_size";
    file_op->create_file("total_statistic.csv", 3);
    file_op->append_file_headers(header);
    std::vector<double> row{0,0,0};

    for (int size = matrix_size_min; size <= matrix_size_max; size += stride) {

        //
        // CPU
        //
        if (s.settings.is_cpu)
        {
            std::unique_ptr<DataInitializerCPU> data = std::make_unique<DataInitializerCPU>(size, 0, size, power);
            std::unique_ptr<NewtonSolverCPU> newton_solver = std::make_unique<NewtonSolverCPU>(data.get());
            newton_solver->cpu_newton_solve();
            row[0] = data->total_elapsed_time;
        }

        //
        // cuBLAS
        //
        if (s.settings.is_cublas)
        {
            std::unique_ptr<DataInitializerCUDA> data2 = std::make_unique<DataInitializerCUDA>(size, 0, size, power);
            std::unique_ptr<NewtonSolverCUDA> newton_solver2 = std::make_unique<NewtonSolverCUDA>(data2.get());
            newton_solver2->gpu_newton_solve();
            row[1] = data2->total_elapsed_time;;
        }

        //
        // cuDSS
        //
        if (s.settings.is_cudss)
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
