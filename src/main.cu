#include <iostream>
#include <time.h>
#include <fstream>
#include <string>
#include <stdlib.h>

#include "NeuralNetwork.hh"
#include "linear_layer.hh"
#include "activation.hh"
#include "softmax.hh"
#include "nodeaggregator.hh"
#include "nn_exception.hh"
#include "costfunction.hh"
#include "csr_graph.h"
#include "data.hh" 

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

        std::fstream myfile("/net/ohm/export/iss/inputs/Learning/cora-labels.txt", std::ios_base::in);
        int* label = (int *) malloc(2708*7*sizeof(int));
        int i = 0;
        int a;
        myfile >> a;
        myfile >> a;
        while (myfile >> a)
        {
            label[i] = a;
            i++;
        }

	srand( time(NULL) );

	//CoordinatesDataset dataset(100, 21);
	CostFunction bce_cost;

//Code for extracting data from dataset files starts here
        CSRGraph graph;
        char gr_file[]="cora.gr";
        char binFile[]="cora-feat.bin";
        int *nnodes = 0,*nedges = 0;
        int feature_size = 1433;
        int label_size = 7;
        graph.read(gr_file,nnodes,nedges);
        int* d_row_start;
        int* d_edge_dst;
        float* d_edge_data;
        cudaError_t alloc;
        int nnz = 5278;
        alloc = cudaMalloc(&d_row_start,(2708+1) * sizeof(*d_row_start));
        if(alloc != cudaSuccess) {
            printf("malloc for row info failed\n");
        }
        alloc = cudaMalloc(&d_edge_dst,(5278) * sizeof(*d_edge_dst));
        if(alloc != cudaSuccess) {
            printf("malloc for col info failed\n");
        }
        float* d_B;

        float* h_B = (float *)malloc((2708) * feature_size * sizeof(float));
	if(h_B == NULL)
	    printf("h_B malloc failed\n");
        alloc = cudaMalloc(&d_B, (2708) * feature_size * sizeof(float));
        if(alloc != cudaSuccess) {
            printf("cudaMalloc failed for features matrix\n");
        }
        alloc = cudaMalloc(&d_edge_data,nnz * sizeof(*d_edge_data));
        if(alloc != cudaSuccess) {
            printf("malloc failed \n");
        }
        alloc = cudaMemset(d_edge_data, 1, 5278*sizeof(float));
        if(alloc != cudaSuccess) {
            printf("memset for edge data failed \n");
        }
//Filling up the sparse matrix info
        graph.readFromGR(gr_file , binFile , d_row_start, d_edge_dst , d_B, feature_size);
        alloc = cudaMemcpy(h_B, d_B, (2708 * 1433 *sizeof(float)), cudaMemcpyDeviceToHost);
	if(alloc != cudaSuccess) {
    	printf("Feature matrix memcpy failed\n");
	} 
	std::cout << "Dataset captured!\n";
        Data dataset(2708,100,feature_size,label_size,label,h_B);
	std::cout << "Dataset captured!\n";
        NeuralNetwork nn;
        std::cout << "Instance of Neural Network\n";
	nn.addLayer(new NodeAggregator("nodeagg1", d_edge_data, d_row_start, d_edge_dst, 2708, nnz));
        std::cout << "Added Nodeaggregator 1 layer\n";
	nn.addLayer(new LinearLayer("linear1", Shape(100,feature_size)));
        std::cout << "Added Linear layer 1\n";
	nn.addLayer(new ReLUActivation("relu2"));
        std::cout << "Added relu layer 1\n";
	nn.addLayer(new NodeAggregator("nodeagg2", d_edge_data, d_row_start, d_edge_dst, 2708, nnz));
        std::cout << "Added Nodeaggregator layer 2\n";
	nn.addLayer(new LinearLayer("linear2", Shape(label_size,100)));
        std::cout << "Added Linear layer 2\n";
	nn.addLayer(new ReLUActivation("relu2"));
        std::cout << "Added Relu layer 2\n";
        nn.addLayer(new SoftMax("softmax"));
        std::cout << "Added softmax layer \n";

        std::cout << "Instance of Neural Network complete\n";
	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < dataset.getNumOfTrainingBatches(); batch++) {
			Y = nn.forward(dataset.getTrainingBatches().at(batch));
			nn.backprop(Y,dataset.getTrainingTargets().at(batch));
			cost += bce_cost.cost(Y,dataset.getTrainingTargets().at(batch));
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / 100
						<< std::endl;
		}
	}

        float accuracy = 0.0f;
        float final_accuracy = 0.0f;
	for (int batch = 0; batch < dataset.getNumOfTestBatches(); batch++) {
		Y = nn.forward(dataset.getTestBatches().at(batch));
		Y.copyDeviceToHost();
                accuracy = accuracy + computeAccuracy(Y,dataset.getTestTargets().at(batch));
	}
        final_accuracy = accuracy/dataset.getNumOfTestBatches();
	// compute accuracy

	std::cout << "Accuracy: " << final_accuracy << std::endl;

	return 0;
}

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
	int m = predictions.shape.x;
	int correct_predictions = 0;

	for (int i = 0; i < m; i++) {
		float prediction = predictions[i] > 0.5 ? 1 : 0;
		if (prediction == targets[i]) {
			correct_predictions++;
		}
	}
	return static_cast<float>(correct_predictions) / m;
}
