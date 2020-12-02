#include <iostream>
#include <time.h>

#include "NeuralNetwork.hh"
#include "linear_layer.hh"
#include "activation.hh"
#include "nodeaggregator.hh"
#include "nn_exception.hh"
#include "costfunction.hh"
#include "csr_graph.h"
float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	srand( time(NULL) );

	//CoordinatesDataset dataset(100, 21);
	CostFunction bce_cost;

//Code for extracting data from dataset files starts here
        CSRGraph graph;
        char gr_file[]="cora.gr";
        char binFile[]="cora-feat.bin";
        int *nnodes = 0,*nedges = 0;
        int feature_size = 1433;
        graph.read(gr_file,nnodes,nedges);
        int* d_row_start;
        int* d_edge_dst;
        float* d_edge_data;
        cudaError_t alloc;
        int nnz = *nedges;
        alloc = cudaMalloc(&d_row_start,(*nnodes+1) * sizeof(*d_row_start));
        if(alloc != cudaSuccess) {
            printf("malloc for row info failed\n");
        }
        alloc = cudaMalloc(&d_edge_dst,(*nedges) * sizeof(*d_edge_dst));
        if(alloc != cudaSuccess) {
            printf("malloc for col info failed\n");
        }
        float* d_B;
        alloc = cudaMalloc(&d_B, (*nnodes) * feature_size * sizeof(float));
        if(alloc != cudaSuccess) {
            printf("cudaMalloc failed for features matrix\n");
        }
        alloc = cudaMalloc(&d_edge_data,nnz * sizeof(*d_edge_data));
        if(alloc != cudaSuccess) {
            printf("malloc failed \n");
        }
        alloc = cudaMemset(d_edge_data, 1, *nedges*sizeof(float));
        if(alloc != cudaSuccess) {
            printf("memset for edge data failed \n");
        }
//Filling up the sparse matrix info
        graph.readFromGR(gr_file , binFile , d_row_start, d_edge_dst , d_B, feature_size);

	NeuralNetwork nn;

	nn.addLayer(new NodeAggregator("nodeagg1", d_edge_data, d_row_start, d_edge_dst, *nnodes, nnz));
	nn.addLayer(new LinearLayer("linear1", Shape(100,20)));
	nn.addLayer(new ReLUActivation("relu2"));
	nn.addLayer(new NodeAggregator("nodeagg2", d_edge_data, d_row_start, d_edge_dst, *nnodes, nnz));
	nn.addLayer(new LinearLayer("linear2", Shape(100,20)));
	nn.addLayer(new ReLUActivation("relu2"));
        nn.addLayer(new SoftMax("softmax"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < 100 - 1; batch++) {
			Y = nn.forward(Y);
			nn.backprop(Y,);
			cost += bce_cost.cost(Y,);
		}

		if (epoch % 100 == 0) {
			std::cout 	<< "Epoch: " << epoch
						<< ", Cost: " << cost / 100
						<< std::endl;
		}
	}

	// compute accuracy
	Y = nn.forward();
	Y.copyDeviceToHost();

	float accuracy = computeAccuracy(Y,);
	std::cout << "Accuracy: " << accuracy << std::endl;

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
