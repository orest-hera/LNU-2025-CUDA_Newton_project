#pragma once

#include <vector>

typedef double(*func)(std::vector<double> el);

extern std::vector<func> funcs;
extern std::vector<double> elements;
