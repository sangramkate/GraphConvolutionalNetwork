#include "data.hh"

Data::Data(size_t batch_size, size_t number_of_batches) :
	batch_size(batch_size), number_of_batches(number_of_batches),feature_size(feature_size),label_size(label_size)
{
	for (int i = 0; i < number_of_batches; i++) {
// adding the input features into a Matrix form and pushing it.
		batches.push_back(Matrix(Shape(batch_size, feature_size)));
		targets.push_back(Matrix(Shape(batch_size, label_size)));

		batches[i].allocateMemory();
		targets[i].allocateMemory();
// Set data under this for loop
		for (int k = 0; k < batch_size; k++) {
		}
// sending the data to gpu for computation
		batches[i].copyHostToDevice();
		targets[i].copyHostToDevice();
	}
}

int Data::getNumOfBatches() {
	return number_of_batches;
}

std::vector<Matrix>& Data::getBatches() {
	return batches;
}

std::vector<Matrix>& Data::getTargets() {
	return targets;
}
