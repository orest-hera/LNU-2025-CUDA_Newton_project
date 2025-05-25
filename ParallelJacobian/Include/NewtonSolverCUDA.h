#pragma once
#include "DataInitializerCUDA.h"

struct NewtonSolverCUDA {
private:
	DataInitializerCUDA* data;

public:
	NewtonSolverCUDA(DataInitializerCUDA* data);
	~NewtonSolverCUDA();

	void gpu_cublasInverse(DataInitializerCUDA* data);
	void gpu_newton_solve();
};