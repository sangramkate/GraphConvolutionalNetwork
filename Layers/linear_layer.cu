#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <random>

#include "linear_layer.hh"
#include "../nn_utils/nn_exception.hh"

__global__ void linearLayerForward( float* W, float* A, float* Z, float* b,
                                                                           int W_x_dim, int W_y_dim,
                                                                           int A_x_dim, int A_y_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    int Z_x_dim = A_x_dim;
    int Z_y_dim = W_y_dim;
  
    float Z_value = 0;
  
    if( row < Z_y_dim && col << Z_x_dim){
       for(int i=0; i< W_x_dim; i=i+1){
           Z_value += W[row * W_x_dim + i] * A[i * A_x_dim + col] 
       }
       Z[row * Z_x_dim + col] = Z_value + b[row];
    }
}

__global__ void linearLayerBackprop( float* W, float* dZ, float*dA,
                                                                    int W_x_dim, int W_y_dim,
                                                                    int dZ_x_dim, int dZ_y_dim){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
  
    int dA_x_dim = dZ_x_dim;
    int dA_y_dim = W_x_dim;
  
    float dA_value = 0.0f;
  	if (row < dA_y_dim && col < dA_x_dim) {
		    for (int i = 0; i < W_y_dim; i++) {
			      dA_value += W[i * W_x_dim + row] * dZ[i * dZ_x_dim + col];
		    }
		    dA[row * dA_x_dim + col] = dA_value;
	  }
}
    
}
