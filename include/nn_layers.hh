#pragma once
#include <iostream>
#include "matrix.hh"

class NNlayer{
protected:
    std::string name;
    
public:
    virtual ~NNLayer() = 0;
    virtual Matrix& forward(Matrix& A) = 0;
    virtual Matrix& backward(Matrix& dZ, float learning rate) = 0;
    
    std::string getName() {return this->name;};
};

inline NNLayer::~NNLayer() {};
