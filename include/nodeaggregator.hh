#pragma once

#include "nn_layer.hh"

class ReLUActivation : public NNLayer {
private:
	Matrix A;
	Matrix Z;
	Matrix dZ;

public:
	NodeAgregator(std::string name,Shape W_shape, float* nnz_data, int* row, int*col, int feature_size);
	~NodeAggregator();

	Matrix& Forward((Matrix& A, float* nnz_data, int* row, int* col));
	Matrix& Backprop(Matrix& dZ, float* nnz_data, int* row, int* col);
};
