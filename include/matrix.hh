#pragma once
#include "shape.hh"
#include <memory>

class Matrix{
private:
    bool device_allocated;
    bool host_allocated;
    
    void allocateCudaMemory();
    void allocateHostMemory();
    
public:
    Shape shape;
    float* data_device;
    float* data_host;
    
    Matrix(size_t x_dim = 1, size_t y_dim = 1);
    Matrix(Shape shape);
    
    void allocateMemory();
    void allocateMemoryIfNotAllocated(Shape shape);
    void copyHostToDevice();
    void copyDeviceToHost();
    void freeMem();    
 
    float& operator[](const int index);
    const float& operator[](const int index) const;
};
