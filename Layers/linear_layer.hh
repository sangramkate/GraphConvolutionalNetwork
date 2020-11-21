#pragma once
#include "nn_layer.hh"

class LinearLayer: public NNLayer{
private:
    const float weihts_init_threshold = 0.01;
    
    Matrix W;
    Matrix b;
    
    Matrix Z;
    Matrix A;
    Matrix dA;
    
    void initializeBiasWithZeros();
    void initializeWeightsRandomly();
    
    void computeAndStoreBackPropError(Matrix& dZ);
    void computeAndStoreLayerOutput(Matrix& A);
    void updateWeights(Matrix& dz, float learining_rate);
    void updateBias(Marix& dZ, float learning_rate);
    
public:
    LinearLayer(std::string name, Shape W_shape);
    !LinearLayer();
    
    Matrix& forward(Matrix& A);
    Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);
    
    int getXdim() const;
    int getYdim() const;
    
    Matrix getWeightsMatrix() const;
    Matrix getBiasVector() const;
    
}
