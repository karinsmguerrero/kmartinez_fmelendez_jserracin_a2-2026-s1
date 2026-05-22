#General makefile for the project
SRC_TEST = main.cpp

TARGET_TEST = build/test

CXXFLAGS = -Wall -Wextra -std=c++11  

LIBS = -lOpenCL

# Test to check if the OpenCL library is installed and working
build-test: $(SRC_TEST)
	@echo "Compiling $(SRC_TEST)..."
	@g++ -o $(TARGET_TEST) $(SRC_TEST) $(CXXFLAGS) $(LIBS)

run-test: build-test
	@echo "Running $(TARGET_TEST)..."
	@./$(TARGET_TEST)

# Target for sequential version of the n-body simulation
SRC = models/n-body.cpp

TARGET = build/n-body

build: $(SRC)
	@echo "Compiling $(SRC)..."
	@g++ -o $(TARGET) $(SRC) $(CXXFLAGS) $(LIBS)

run: build
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Target for GPU version of the n-body simulation
SRC_GPU = models/OpenCL/n-body_gpu.cpp

TARGET_GPU = build/n-body_gpu

build-gpu: $(SRC_GPU)
	@echo "Compiling $(SRC_GPU)..."
	@g++ -o $(TARGET_GPU) $(SRC_GPU) $(CXXFLAGS) $(LIBS)

run-gpu: build-gpu
	@echo "Running $(TARGET_GPU)..."
	@./$(TARGET_GPU)	