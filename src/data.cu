#include "data.hh"

Data::Data(int num_nodes, size_t batch_size,int feature_size, int label_size, int* label, float* feature) :
	num_nodes(num_nodes),batch_size(batch_size),feature_size(feature_size),label_size(label_size),feature(feature), label(label)
{
        number_of_batches = (int) (num_nodes/batch_size);
        num_training_batches = (int) (number_of_batches * 0.4);
        num_test_batches = number_of_batches- num_training_batches;
	for (int i = 0; i < num_training_batches+1; i++) {
// adding the input features into a Matrix form and pushing it.
	    training_batches.push_back(Matrix(Shape(batch_size, feature_size)));
	    training_targets.push_back(Matrix(Shape(batch_size, label_size)));
	    training_batches[i].allocateMemory();
	    training_targets[i].allocateMemory();
// Set data under this for loop
            for(int j=0; j < batch_size; j++ ){
                for(int k=0; k < feature_size; k++) {
                    training_batches[i][j] = feature[ k + feature_size * j  + batch_size * feature_size * i ];
                }
            }
            for(int j=0; j < label_size; j++){
                for(int k=0; k < feature_size; k++) {
                    training_batches[i][j] = label[ k + label_size * j  + batch_size * label_size * i ];
                }
            }  
// sending the data to gpu for computation
	    training_batches[i].copyHostToDevice();
	    training_targets[i].copyHostToDevice();
	}
	for (int i = 0; i < num_test_batches+1; i++) {
// adding the input features into a Matrix form and pushing it.
	    test_batches.push_back(Matrix(Shape(batch_size, feature_size)));
	    test_targets.push_back(Matrix(Shape(batch_size, label_size)));
	    test_batches[i].allocateMemory();
	    test_targets[i].allocateMemory();
// Set data under this for loop
            for(int j=0; j < batch_size; j++ ){
                for(int k=0; k < feature_size; k++) {
                    test_batches[i][j] = feature[ k + feature_size * j  + batch_size * feature_size * i ];
                }
            }
            for(int j=0; j < label_size; j++){
                for(int k=0; k < feature_size; k++) {
                    test_batches[i][j] = label[ k + label_size * j  + batch_size * label_size * i ];
                }
            }  
// sending the data to gpu for computation
	    test_batches[i].copyHostToDevice();
	    test_targets[i].copyHostToDevice();
	}
}

int Data::getNumOfTrainingBatches() {
	return num_training_batches;
}

int Data::getNumOfTestBatches() {
	return num_test_batches;
}

std::vector<Matrix>& Data::getTrainingBatches() {
	return training_batches;
}

std::vector<Matrix>& Data::getTestBatches() {
	return test_batches;
}

std::vector<Matrix>& Data::getTrainingTargets() {
	return training_targets;
}

std::vector<Matrix>& Data::getTestTargets() {
	return test_targets;
}
