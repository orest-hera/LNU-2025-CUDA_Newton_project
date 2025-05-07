#pragma once

class Equation {
private:
	int power;
public:
	Equation(int power);
	double calculate_term_value(double index, double x);
	int get_power();
};