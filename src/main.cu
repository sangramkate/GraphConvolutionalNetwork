#include <iostream>
#include <time.h>

#include "NeuralNetwork.hh"
#include "linear_layer.hh"
#include "activation.hh"
#include "nodeaggregator.hh"
#include "nn_exception.hh"
#include "costfunction.hh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets);

int main() {

	srand( time(NULL) );

	//CoordinatesDataset dataset(100, 21);
	CostFunction bce_cost;

	NeuralNetwork nn;
	nn.addLayer(new LinearLayer("linear1", Shape(100,20)));
	nn.addLayer(new ReLUActivation("relu2"));
	nn.addLayer(new LinearLayer("linear2", Shape(100,20)));
	nn.addLayer(new ReLUActivation("relu2"));

	// network training
	Matrix Y;
	for (int epoch = 0; epoch < 1001; epoch++) {
		float cost = 0.0;

		for (int batch = 0; batch < 100 - 1; batch++) {
			Y = nn.forward();
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
