#include "costfunction.hh"
#include "nn_exception.hh"

#include <math.h>
#include <iostream>
#include <assert.h>

__global__ void binaryCrossEntropyCost(float* predictions, float* target,
									   int size, float* cost) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		float partial_cost = target[index] * logf(predictions[index])
				+ (1.0f - target[index]) * logf(1.0f - predictions[index]);
		atomicAdd(cost, - partial_cost / size);
	}
}

__global__ void dBinaryCrossEntropyCost(float* predictions, float* target, float* dY,
								     	int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < size) {
		dY[index] = -1.0 * ( target[index]/predictions[index] - (1 - target[index])/(1 - predictions[index]) );
	}
}

float CostFunction::cost(Matrix predictions, Matrix target) {
      //  std::cout << "predictions.x:" << predictions.shape.x <<"\n" ;
      //  std::cout << "predictions.y:" << predictions.shape.y <<"\n" ;
      //  std::cout << "target.x:" << target.shape.x <<"\n" ;
      //  std::cout << "target.y:" << target.shape.y <<"\n" ;
	assert(predictions.shape.y == target.shape.y);

	float* cost;
     //   std:: cout << "pointer created\n";
	cudaMallocManaged(&cost, sizeof(float));
     //   std::cout <<"this gets printed\n";
     //   std:: cout << "Memory Allocated\n";
	*cost = 0.0f;
     //   std:: cout << "cost initialized\nn";

	dim3 block_size(256);
      // std:: cout << "dim3 block size\nn";
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
      //  std::cout << "start finding cross entropy\n";
	binaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device, target.data_device,predictions.shape.x, cost);
      //  std::cout << "done finding cross entropy\n";
	cudaDeviceSynchronize();
	NNException::throwIfDeviceErrorOccurred("Cannot compute binary cross entropy cost.");

	float cost_value = *cost;
	cudaFree(cost);

	return cost_value;
}

Matrix CostFunction::dCost(Matrix predictions, Matrix target, Matrix dY) {
	assert(predictions.shape.x == target.shape.x);

	dim3 block_size(256);
	dim3 num_of_blocks((predictions.shape.x + block_size.x - 1) / block_size.x);
	dBinaryCrossEntropyCost<<<num_of_blocks, block_size>>>(predictions.data_device,
														   target.data_device,
														   dY.data_device,
														   predictions.shape.x);
	NNException::throwIfDeviceErrorOccurred("Cannot compute derivative for binary cross entropy.");

	return dY;
}
