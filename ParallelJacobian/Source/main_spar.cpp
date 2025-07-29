#include <iostream>
#include <memory>
#include <vector>

#include "FileOperations.h"
#include "NewtonSolverCPU.h"
#include "NewtonSolverCUDA.h"
#include "NewtonSolverCuDSS.h"
#include "config.h"
#include "settings.h"
#include "system-build-info.h"

int main(int argc, char* argv[]) {
    SystemBuildInfo::dump(std::cout);
    std::cout << std::endl;

    Settings s;

    int matrix_size = 1000;

    int zeros_per_row_max = 995;
    int zeros_per_row_min = 995;
    int stride = 100;
    int power = 3;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];

        if (arg.find("--power=") == 0) {
            power = std::stod(arg.substr(8));
        }
    }
    if (argc > 1) {
        zeros_per_row_max = std::atoi(argv[1]);

        if (argc > 2) {
            zeros_per_row_min = std::atoi(argv[2]);

            if (argc > 3) {
                stride = std::atoi(argv[3]);
            }
        }
    }

    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(s.settings.path);
    std::string header = "CPU,GPU,cuDSS,zeros_per_row";
    file_op->create_file("total_statistic.csv", 3);
    file_op->append_file_headers(header);
    std::vector<double> row(3);

    for (int size = zeros_per_row_min; size <= zeros_per_row_max; size += stride) {

        //
        // CPY
        //
        {
            std::unique_ptr<DataInitializerCPU> data = std::make_unique<DataInitializerCPU>(matrix_size, size, size, power);
            auto newton_solver = std::make_unique<NewtonSolverCPU>(data.get(), s.settings);
            newton_solver->cpu_newton_solve();
            row[0] = data->total_elapsed_time;
        }
        ////
        //// GPU
        ////
        {
            std::unique_ptr<DataInitializerCUDA> data2 = std::make_unique<DataInitializerCUDA>(matrix_size, size, size, power);
            auto newton_solver2 = std::make_unique<NewtonSolverCUDA>(data2.get(), s.settings);
            newton_solver2->gpu_newton_solve();
            row[1] = data2->total_elapsed_time;
        }
        //
        // cuDSS
        //
        {
            std::unique_ptr<DataInitializerCuDSS> data3 = std::make_unique<DataInitializerCuDSS>(matrix_size, size, size, power);
            auto cuDssSolver = std::make_unique<NewtonSolverCuDSS>(data3.get(), s.settings);
            cuDssSolver->gpu_newton_solver_cudss();
            row[2] = data3->total_elapsed_time;
        }
        file_op->append_file_data(row, size);
    }
    return 0;
}
