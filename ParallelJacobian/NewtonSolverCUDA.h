#pragma once

#include "DataInitializer.h"

struct NewtonSolverCUDA {
private:
	DataInitializer* data;

public:
	NewtonSolverCUDA(DataInitializer* data);
	~NewtonSolverCUDA();

	void gpu_cublasInverse(DataInitializer* data);
	void gpu_newton_solve();
};