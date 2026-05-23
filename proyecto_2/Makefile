#General makefile for the project
SRC_TEST = main.cpp

TARGET_TEST = build/test

CXXFLAGS = -Wall -Wextra -std=c++11  

PYTHON ?= python3
VISUALIZE = scripts/visualize.py
VENV ?= .venv
VENV_PY = $(VENV)/bin/python
REQS = requirements.txt

CXXDEFS =
ifdef NBODIES
CXXDEFS += -DNBODIES=$(NBODIES)
endif
ifdef NSTEPS
CXXDEFS += -DNSTEPS=$(NSTEPS)
endif
ifdef NBSAVEFRAMES
CXXDEFS += -DNBSAVEFRAMES=$(NBSAVEFRAMES)
endif
ifdef NBVERBOSE
CXXDEFS += -DNBVERBOSE=$(NBVERBOSE)
endif

LIBS = -lOpenCL

# Test to check if the OpenCL library is installed and working
build-test: $(SRC_TEST)
	@echo "Compiling $(SRC_TEST)..."
	@mkdir -p build
	@g++ -o $(TARGET_TEST) $(SRC_TEST) $(CXXFLAGS) $(LIBS)

run-test: build-test
	@echo "Running $(TARGET_TEST)..."
	@./$(TARGET_TEST)

# Target for sequential version of the n-body simulation
SRC = models/n-body.cpp

TARGET = build/n-body

build: $(SRC)
	@echo "Compiling $(SRC)..."
	@mkdir -p build
	@g++ -o $(TARGET) $(SRC) $(CXXFLAGS) $(CXXDEFS) $(LIBS)

run: build
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Target for GPU version of the n-body simulation
SRC_GPU = models/OpenCL/n-body_gpu.cpp

TARGET_GPU = build/n-body_gpu

build-gpu: $(SRC_GPU)
	@echo "Compiling $(SRC_GPU)..."
	@mkdir -p build
	@g++ -o $(TARGET_GPU) $(SRC_GPU) $(CXXFLAGS) $(CXXDEFS) $(LIBS)

run-gpu: build-gpu
	@echo "Running $(TARGET_GPU)..."
	@./$(TARGET_GPU)	

# Target to visualize saved frames
run-visualize:
	@echo "Running $(VISUALIZE)..."
	@$(PYTHON) $(VISUALIZE)

# Target to set up venv for visualization
setup-venv:
	@echo "Setting up venv in $(VENV)..."
	@$(PYTHON) -m venv $(VENV)
	@$(VENV_PY) -m pip install --upgrade pip
	@$(VENV_PY) -m pip install -r $(REQS)

# Target to visualize saved frames using venv
run-visualize:
	@echo "Running $(VISUALIZE) with venv..."
	@$(VENV_PY) $(VISUALIZE)