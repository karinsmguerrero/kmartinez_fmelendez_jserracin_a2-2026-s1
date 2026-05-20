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

SRC = n-body.cpp

TARGET = build/n-body

build: $(SRC)
	@echo "Compiling $(SRC)..."
	@g++ -o $(TARGET) $(SRC) $(CXXFLAGS) $(LIBS)

run: build
	@echo "Running $(TARGET)..."
	@./$(TARGET)