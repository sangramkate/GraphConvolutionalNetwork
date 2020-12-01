#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include "src/nn_utils/nn_exception.hh"
#include "csr_graph.cu"
#include "csr_graph.h"

__global__ void NodeAggregatorForward(float* nnz_data, int* row, int* col, float* A, float* Z, int feature_size) {

	CSRGraph C;
	C.SpMM(*nnz_data, *row, *col, A, feature_size, Z);
}

__global__ void NodeAggregatorBackProp(float* nnz_data, int* row, int* col, float* dZ, float* dA, int feature_size) {

	CSRGraph C;
	C.SpMM(*nnz_data, *row, *col, dZ, feature_size, dA);
}

Matrix& NodeAggregator::Forward(Matrix& A){
this->A = A;
Z.allocateMemoryIfNotAllocated(A.shape);
dim3 block_size(256);
dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
NodeAggregatorForward<<<num_of_blocks, block_size>>>(nnz_data, row, col, A.data_device.get(), Z.data_device.get(), A.shape.x);
return Z;
}

Matrix& NodeAggregator::BackProp(Matrix& dZ) {
this->dZ = dZ;
dA.allocateMemoryIfNotAllocated(dZ.shape);
dim3 block_size(256);
dim3 num_of_blocks((dA.shape.y * dA.shape.x + block_size.x - 1) / block_size.x);
NodeAggregatorBackProp<<<num_of_blocks, block_size>>>(nnz_data, row, col, dZ.data_device.get(), dA.data_device.get(), dZ.shape.x);
return dA;
}

NodeAggregator::NodeAggregator(std::string name, float* nnz_data, int* row, int*col):
{
    this->name = name;
    this->nnz_data = nnz_data;
    this->row = row;
    this->col = col;
}

NodeAggregator::~NodeAggregator() { }
