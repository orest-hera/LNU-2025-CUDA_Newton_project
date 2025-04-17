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
    //
    // CPY
    //
#ifdef CPU_SOLVER
    {
        std::unique_ptr<DataInitializer> data = std::make_unique<DataInitializer>(1000);
        std::unique_ptr<NewtonSolver> newton_solver = std::make_unique<NewtonSolver>(data.get());
        newton_solver->cpu_newton_solve();
    }
#endif

    //
    // GPU
    //
#ifdef GPU_SOLVER
    {
        std::unique_ptr<DataInitializer> data2 = std::make_unique<DataInitializer>(1000);
        std::unique_ptr<NewtonSolver> newton_solver2 = std::make_unique<NewtonSolver>(data2.get());
        newton_solver2->gpu_newton_solve();
    }
#endif
#ifdef CUDSS_SOLVER
    {
        std::unique_ptr<CuDssSolver> cuDssSolver = std::make_unique<CuDssSolver>(1000);
        cuDssSolver->solve();
    }
#endif
    return 0;
}