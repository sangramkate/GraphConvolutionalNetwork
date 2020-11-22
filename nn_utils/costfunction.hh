#pragma once
#include "matrix.hh"

class costfunction {
public:
	float cost(Matrix predictions, Matrix target);
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
