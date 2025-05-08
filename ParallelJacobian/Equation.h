#pragma once
#include <vector>
#include <cmath>

class Equation {
private:
	int power;
public:
	Equation(int power);
	double calculate_term_value(double index, double x);
	int get_power();
};