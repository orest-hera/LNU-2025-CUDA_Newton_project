#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "iostream"
#include "math.h"
#include "vector"
#include "DataInitializer.h"
#include "NewtonSolver.h"
#include "memory"
#include "config.h"
#include "CuDssSolver.h"
#include <cstdlib>

int main(int argc, char* argv[]) {

    int matrix_size_max = 100;
	int matrix_size_min = 100;
	int stride = 100;
    if (argc > 1) {
        matrix_size_max = std::atoi(argv[1]);

        if (argc > 2) {
            matrix_size_min = std::atoi(argv[2]);

            if (argc > 3) {
                stride = std::atoi(argv[3]);
            }
        }
    }

    for (int size = matrix_size_min; size <= matrix_size_max; size += stride) {
        //
        // CPY
        //
#ifdef CPU_SOLVER
        {
            std::unique_ptr<DataInitializer> data = std::make_unique<DataInitializer>(size);
            std::unique_ptr<NewtonSolver> newton_solver = std::make_unique<NewtonSolver>(data.get());
            newton_solver->cpu_newton_solve();
        }
#endif

        //
        // GPU
        //
#ifdef GPU_SOLVER
        {
            std::unique_ptr<DataInitializer> data2 = std::make_unique<DataInitializer>(size);
            std::unique_ptr<NewtonSolver> newton_solver2 = std::make_unique<NewtonSolver>(data2.get());
            newton_solver2->gpu_newton_solve();
        }
#endif
#ifdef CUDSS_SOLVER
        {
            std::unique_ptr<CuDssSolver> cuDssSolver = std::make_unique<CuDssSolver>(size);
            cuDssSolver->solve();
        }
#endif
    }
    return 0;
}