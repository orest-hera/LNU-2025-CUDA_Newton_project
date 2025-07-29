#pragma once

#include <chrono>

#include "DataInitializerCPU.h"
#include "config.h"
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
