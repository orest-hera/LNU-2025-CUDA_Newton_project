#pragma once
#include "DataInitializerCUDA.h"

#include "settings.h"

struct NewtonSolverCUDA {
private:
	DataInitializerCUDA* data;
	const Settings::SettingsData& settings_;

public:
	NewtonSolverCUDA(DataInitializerCUDA* data, const Settings::SettingsData& settings);
	~NewtonSolverCUDA();

	void gpu_cublasInverse(DataInitializerCUDA* data);
	void gpu_newton_solve();
};
