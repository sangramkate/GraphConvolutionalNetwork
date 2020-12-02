#pragma once

#include "matrix.hh"

#include <vector>

class Data {
private:
	size_t batch_size;
	size_t number_of_batches;
        size_t feature_size;
        size_t label_size;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	Data(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};
