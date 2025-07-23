#pragma once
#include "DataInitializerCPU.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "config.h"
#include <chrono>

#include "settings.h"

struct NewtonSolverCPU {
private:
	DataInitializerCPU * data;
	const Settings::SettingsData& settings_;

	void cpu_computeVec();
	double cpu_compute_derivative(int rowIndex, int colIndex);
	void cpu_compute_jacobian();
	void cpu_inverse();
	void cpu_compute_delta();

public:
	NewtonSolverCPU(DataInitializerCPU* data, const Settings::SettingsData& settings);
	~NewtonSolverCPU();

	void cpu_newton_solve();
};
