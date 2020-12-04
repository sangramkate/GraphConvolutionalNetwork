#include "activation.hh"
#include "nn_exception.hh"

__global__ void ReluActivationForward(float* Z, float* A, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		A[index] = fmaxf(Z[index], 0);
	}
}

__global__ void ReluActivationBackprop(float* Z, float* dA, float* dZ, int Z_x_dim, int Z_y_dim) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < Z_x_dim * Z_y_dim) {
		if (Z[index] > 0) {
			dZ[index] = dA[index];
		}
		else {
			dZ[index] = 0;
		}
	}
}

ReLUActivation::ReLUActivation(std::string name) {
	this->name = name;
}

ReLUActivation::~ReLUActivation() { }

Matrix& ReLUActivation::forward(Matrix& Z) {
	this->Z = Z;
	A.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);

	ReluActivationForward<<<num_of_blocks, block_size>>>(Z.data_device, A.data_device, Z.shape.x, Z.shape.y);
        std::cout << "Relu forward\n";
	NNException::throwIfDeviceErrorOccurred("Cannot perform ReLU forward propagation.");
        std::cout << " Relu forward shape.x:" << A.shape.x << "\n";
        std::cout << " Relu forward shape.y:" << A.shape.y << "\n";
        Z.freeMem();
	return A;
}

Matrix& ReLUActivation::backprop(Matrix& dA, float learning_rate) {
	dZ.allocateMemoryIfNotAllocated(Z.shape);

	dim3 block_size(256);
	dim3 num_of_blocks((Z.shape.y * Z.shape.x + block_size.x - 1) / block_size.x);
	ReluActivationBackprop<<<num_of_blocks, block_size>>>(Z.data_device, dA.data_device,dZ.data_device, Z.shape.x, Z.shape.y);
        std::cout << "Relu Backward\n"; 	
        NNException::throwIfDeviceErrorOccurred("Cannot perform ReLU back propagation");

        std::cout << " Relu backward shape.x:" << dZ.shape.x << "\n";
        std::cout << " Relu backward shape.y:" << dZ.shape.y << "\n";
        dA.freeMem();
	return dZ;
}
