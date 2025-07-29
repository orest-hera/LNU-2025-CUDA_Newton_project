#include <iostream>
#include <memory>
#include <vector>

#include "FileOperations.h"
#include "NewtonSolverCPU.h"
#ifdef CFG_SOLVE_CUDA
#include "NewtonSolverCUDA.h"
#include "NewtonSolverCuDSS.h"
#endif
#ifdef CFG_SOLVE_MKL
#include "NewtonSolverMKLdss.h"
#include "NewtonSolverMKLlapack.h"
#endif
#include "config.h"
#include "report.h"
#include "settings.h"
#include "system-build-info.h"
#include "system-info.h"

int main(int argc, char* argv[]) {
    SystemBuildInfo::dump(std::cout);
    std::cout << std::endl;

    Settings s;
    if (!s.parse(argc, argv))
        return 1;

    SystemInfo sinfo(argc, argv);

    if (s.settings.report_subdir) {
        std::string path = s.settings.path + "/" + sinfo.getTimeStamp();
        if (!Report::createReportDir(path))
            return 2;

        s.settings.path = path;
    }

    Report::RedirectOut rdout;
    if (s.settings.redirect_out) {
        rdout.redirect(s.settings.path);

        SystemBuildInfo::dump(std::cout);
        std::cout << std::endl;
    }

    sinfo.dump(std::cout);

    unsigned matrix_size_max = s.settings.max;
    unsigned matrix_size_min = s.settings.min;
    int stride = s.settings.stride;
    int power = s.settings.power;

    if (s.settings.size != 0) {
        // use only one matrix size
        matrix_size_max = s.settings.size;
        matrix_size_min = s.settings.size;
    }

    std::unique_ptr<FileOperations> file_op = std::make_unique<FileOperations>(s.settings.path);
    std::string header = "CPU,GPU,cuDSS,MKL_Lapack,MKL_DSS,matrix_size";
    file_op->create_file("total_statistic.csv", 5);
    file_op->append_file_headers(header);
    std::vector<double> row{0,0,0,0,0};

    for (unsigned size = matrix_size_min; size <= matrix_size_max; size += stride) {
        unsigned num_zeros = s.settings.nnz != 0 && s.settings.nnz < size ?
                    size - s.settings.nnz : 0;

        //
        // CPU
        //
        if (s.settings.is_cpu)
        {
            auto data = std::make_unique<DataInitializerCPU>(size, num_zeros, size, power);
            auto newton_solver = std::make_unique<NewtonSolverCPU>(data.get(), s.settings);
            newton_solver->cpu_newton_solve();
            row[0] = data->total_elapsed_time;
        }

#ifdef CFG_SOLVE_CUDA
        //
        // cuBLAS
        //
        if (s.settings.is_cublas)
        {
            auto data2 = std::make_unique<DataInitializerCUDA>(size, num_zeros, size, power);
            auto newton_solver2 = std::make_unique<NewtonSolverCUDA>(data2.get(), s.settings);
            newton_solver2->gpu_newton_solve();
            row[1] = data2->total_elapsed_time;;
        }

        //
        // cuDSS
        //
        if (s.settings.is_cudss)
        {
            auto data3 = std::make_unique<DataInitializerCuDSS>(size, num_zeros, size, power);
            auto cuDssSolver = std::make_unique<NewtonSolverCuDSS>(data3.get(), s.settings);
            cuDssSolver->gpu_newton_solver_cudss();
            row[2] = data3->total_elapsed_time;;
        }
#endif
#ifdef CFG_SOLVE_MKL
        //
        // MKL Lapack
        //
        if (s.settings.is_mkl_lapack)
        {
            auto data = std::make_unique<DataInitializerMKLlapack>(size, num_zeros, size, power);
            auto mklLapackSolver = std::make_unique<NewtonSolverMKLlapack>(data.get(), s.settings);
            mklLapackSolver->cpu_newton_solve();
            row[3] = data->total_elapsed_time;;
        }

        //
        // MKL DSS
        //
        if (s.settings.is_mkl_dss)
        {
            auto data = std::make_unique<DataInitializerMKLdss>(size, num_zeros, size, power);
            auto mklDssSolver = std::make_unique<NewtonSolverMKLdss>(data.get(), s.settings);
            mklDssSolver->cpu_newton_solve();
            row[4] = data->total_elapsed_time;;
        }
#endif

        file_op->append_file_data(row, size);
    }
    return 0;
}
