#pragma once
#include "DataInitializerMKLlapack.h"

#include "settings.h"

struct NewtonSolverMKLlapack
{
private:
    DataInitializerMKLlapack * data;
    const Settings::SettingsData& settings_;

    void cpu_computeVec();
	double cpu_compute_derivative(int rowIndex, int colIndex);
    void cpu_compute_jacobian();
    void cpu_find_delta();

public:
    NewtonSolverMKLlapack(DataInitializerMKLlapack* data, const Settings::SettingsData& settings);
    ~NewtonSolverMKLlapack();
    void cpu_newton_solve();
};
