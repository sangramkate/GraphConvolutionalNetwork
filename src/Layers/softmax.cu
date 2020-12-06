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
                        dA[j* dA_y_dim + col] += dZ[j * dA_y_dim + i] * (sum - exp(dZ[j * dA_y_dim + i]))/ (sum * sum) * exp(dZ[j * dA_x_dim + i]);
                    }
                    else{
                        dA[j* dA_x_dim + col] -= dZ[j * dA_x_dim + i] *  exp(dZ[j * dA_x_dim + i])/ (sum * sum) * exp(dZ[j * dA_x_dim + i]);
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

Matrix& SoftMax::forward(Matrix& A,bool  training, bool freeMatrix){
    this->A = A;
    Shape Z_shape(A.shape.x,A.shape.y);
    Z.allocateCuda(Z_shape);
    std::cout<<"softmax forward\n";
    LayerOutput(A);
    NNException::throwIfDeviceErrorOccurred("Cannot perform Linear Layer forward propagation");
    if(freeMatrix)
        A.freeMem();
    return Z;
}
void SoftMax::LayerOutput(Matrix& A) {
    int block_size(256);
    int num_of_blocks((Z.shape.x + block_size - 1) / block_size);
    SoftMaxForward<<<num_of_blocks, block_size>>>( A.data_device,Z.data_device,A.shape.x, A.shape.y);
}

Matrix& SoftMax::backprop(Matrix& dZ, float learning_rate) {
    dA.allocateCuda(A.shape);
    std::cout<<"softmax backward\n";
    BackpropError(dZ);
    NNException::throwIfDeviceErrorOccurred("Cannot perform back propagation.");
    //dZ.freeMem();
    return dA;
}

void SoftMax::BackpropError(Matrix& dZ) {
    int block_size(256);
    int num_of_blocks ((dZ.shape.x + block_size - 1) / block_size);
    SoftMaxBackprop<<<num_of_blocks, block_size >>> ( dZ.data_device,dA.data_device,dZ.shape.x, dZ.shape.y);
}
