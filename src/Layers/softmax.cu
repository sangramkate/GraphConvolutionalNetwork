#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "softmax.hh"
#include "nn_exception.hh"

__global__ void SoftMaxForward( float* A, float* Z,int A_x_dim, int A_y_dim){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    int Z_x_dim = A_x_dim;
    int Z_y_dim = A_y_dim;
  
    float sum = 0.0f;
    
    if(col << Z_x_dim){
       for(int i=0; i< Z_y_dim; i=i+1){
           float tmp = exp(A[i * Z_x_dim + col]);
           Z[i* Z_x_dim + col] = tmp;
           sum += tmp;  
       }
       for(int i= 0; i < Z_y_dim; i=i+1){
           Z[i * Z_x_dim + col] /= sum;
       }
    }
}

__global__ void SoftMaxBackprop( float* dZ, float*dA, int dZ_x_dim, int dZ_y_dim){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = dZ_y_dim;
  
    float sum = 0.0f;
  	if (col < dA_x_dim) {
            for(int i=0; i< dA_y_dim; i=i+1){
                float tmp = exp(dZ[i * dA_x_dim + col]);
                dA[i* dA_x_dim + col] = tmp;
                sum += tmp;  
            }
            for(int j=0; j< dA_y_dim; j=j+1){
                for(int i=0; i< dA_y_dim; i=i+1){
                    if(i==j){
                        dA[i* dA_x_dim + col] += dZ[i * dA_x_dim + j] * (sum - exp(dZ[i * dA_x_dim + i]))/ (sum * sum) * exp(dZ[i * dA_x_dim + i]);
                    }
                    else{
                        dA[i* dA_x_dim + col] -= dZ[i * dA_x_dim + j] *  exp(dZ[i * dA_x_dim + i])/ (sum * sum) * exp(dZ[i * dA_x_dim + i]);
                    }
                }
            }
	  }
}

SoftMax::SoftMax(std::string name)
{
    this->name = name;
}

SoftMax::~SoftMax()
{ }

Matrix& SoftMax::forward(Matrix& A){
    this->A = A;
    Shape Z_shape(A.shape.x,A.shape.y);
    Z.allocateMemoryIfNotAllocated(Z_shape);
    std::cout<<"softmax forward\n";
    LayerOutput(A);
    NNException::throwIfDeviceErrorOccurred("Cannot perform Linear Layer forward propagation");
    return Z;
}
void SoftMax::LayerOutput(Matrix& A) {
    int block_size(256);
    int num_of_blocks((Z.shape.x + block_size - 1) / block_size);
    SoftMaxForward<<<num_of_blocks, block_size>>>( A.data_device.get(),Z.data_device.get(),A.shape.x, A.shape.y);
}

Matrix& SoftMax::backprop(Matrix& dZ, float learning_rate) {
    dA.allocateMemoryIfNotAllocated(A.shape);
    std::cout<<"softmax backward\n";
    BackpropError(dZ);
    NNException::throwIfDeviceErrorOccurred("Cannot perform back propagation.");
    return dA;
}

void SoftMax::BackpropError(Matrix& dZ) {
    int block_size(256);
    int num_of_blocks ((dZ.shape.x + block_size - 1) / block_size);
    SoftMaxBackprop<<<num_of_blocks, block_size >>> ( dZ.data_device.get(),dA.data_device.get(),dZ.shape.x, dZ.shape.y);
}