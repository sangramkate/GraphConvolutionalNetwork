#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>
#include "nodeaggregator.hh"
#include "csr_graph.cu"
#include "csr_graph.h"

Matrix& NodeAggregator::forward(Matrix& A){
this->A = A;
Z.allocateMemoryIfNotAllocated(A.shape);
SpMM(nnz_data, row, col, A.data_device.get(), Z.data_device.get(), A.shape.x, nodes, nnz);
return Z;
}

Matrix& NodeAggregator::backprop(Matrix& dZ, float learning_rate) {
this->dZ = dZ;
dA.allocateMemoryIfNotAllocated(dZ.shape);
SpMM(nnz_data, row, col, dZ.data_device.get(), dA.data_device.get(), A.shape.x, nodes, nnz);
return dA;
}

NodeAggregator::NodeAggregator(std::string name, float* nnz_data, int* row, int*col, int nodes, int nnz)
{
    this->name = name;
    this->nnz_data = nnz_data;
    this->row = row;
    this->col = col;
    this->nodes = nodes;
    this->nnz = nnz;
}

NodeAggregator::~NodeAggregator() { }
