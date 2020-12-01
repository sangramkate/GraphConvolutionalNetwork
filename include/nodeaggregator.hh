#pragma once
#include "nn_layers.hh"

class NodeAggregator: public NNLayer{
private:
    Matrix Z;
    Matrix A;
    Matrix dZ;
    Matrix dA;
    float* nnz_data;
    int* row;
    int* col;
    int nodes;
    int nnz;
    
public:
    NodeAggregator(std::string name, float* nnz_data, int* row, int*col, int nodes, int nnz);
    ~NodeAggregator();
    
    Matrix& Forward(Matrix& A);
    Matrix& BackProp(Matrix& dZ);
   
//    int getXdim() const;
//    int getYdim() const;
    
};
