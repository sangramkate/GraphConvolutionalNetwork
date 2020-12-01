###########################################################
## USER SPECIFIC DIRECTORIES ##
# CUDA directory:

CUDA_ROOT_DIR=/usr/local/cuda

##########################################################
## CC COMPILER OPTIONS ##
# CC compiler options:

CC=g++
CC_FLAGS=
CC_LIBS=

##########################################################
## NVCC COMPILER OPTIONS ##
# NVCC compiler options:

NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# CUDA library directory:

CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:

CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:

CUDA_LINK_LIBS= -lcudart

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file diretory:
INC_DIR = include/
INC_FILES = $(INC_DIR)/activation.hh,$(INC_DIR)/costfunction.hh,$(INC_DIR)/linear_layer.hh,$(INC_DIR)/matrix.hh,$(INC_DIR)/NeuralNetwork.hh,$(INC_DIR)/nn_exception.hh,$(INC_DIR)/nn_layers.hh,$(INC_DIR)/nodeaggregator.hh,$(INC_DIR)/shape.hh
##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
OBJS = $(OBJ_DIR)/main.o $(OBJ_DIR)/cuda_kernel.o

.SUFFIXES: .cu .o
##########################################################

## Compile ##

# Clean objects in object directory.

# Link c++ and CUDA compiled object files to target executable:
#$(EXE) : $(OBJ_DIR)/%.o
#	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_LINK_LIBS)

#  Compile main .cpp file to object files:
# $(OBJ_DIR)/%.o : %.cpp
#	$(CC) $(CC_FLAGS) -c $< -o $@

#  Compile C++ source files to object files:
# $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu ${INC_DIR}/%.h ${SRC_DIR}/layers/%.cu ${SRC_DIR}/nn_utils/%.cu
#	$(CC) $(CC_FLAGS) -c $< -o $@
# Compile CUDA source files to object files:

clean:
	$(RM) -rf bin
	mkdir bin

makeobj:
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/Layers/activation.cu -o $(OBJ_DIR)/activation.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/Layers/linear_layer.cu -o $(OBJ_DIR)/linear_layer.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/Layers/nodeaggregator.cu -o $(OBJ_DIR)/nodeaggregator.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/nn_utils/costfunction.cu -o $(OBJ_DIR)/costfunction.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/nn_utils/matrix.cu -o $(OBJ_DIR)/shape.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/nn_utils/shape.cu -o $(OBJ_DIR)/shape.o 
	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/NeuralNetwork.cu -o $(OBJ_DIR)/NeuralNetwork.o 
#	$(NVCC) $(NVCC_FLAGS) -std=c++11  -I $(INC_DIR) -c src/main.cu -o $(OBJ_DIR)/main.o 

all: clean makeobj
