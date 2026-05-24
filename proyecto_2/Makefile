#General makefile for the project
CXX ?= g++
CXXFLAGS = -Wall -Wextra -std=c++11 -ffp-contract=off -fno-fast-math -fexcess-precision=standard

PYTHON ?= python3
VISUALIZE = scripts/visualize.py
VALIDATE = scripts/validation.py
VENV ?= .venv
VENV_PY = $(VENV)/bin/python
REQS = requirements.txt
COMPARE_ARGS ?=

NBODIES ?= 1000
NSTEPS ?= 100
NBSAVEFRAMES ?= 1
NBVERBOSE ?= 0

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

BUILD_DIR = build
LIBS_GPU = -lOpenCL

# Target for sequential version of the n-body simulation
SRC = models/n-body.cpp

TARGET = $(BUILD_DIR)/n-body

build: $(SRC)
	@echo "Compiling $(SRC)..."
	@mkdir -p $(BUILD_DIR)
	@$(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS) $(CXXDEFS)

run: build
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Target for GPU version of the n-body simulation
SRC_GPU = models/OpenCL/n-body_gpu.cpp

TARGET_GPU = $(BUILD_DIR)/n-body_gpu

build-gpu: $(SRC_GPU)
	@echo "Compiling $(SRC_GPU)..."
	@mkdir -p $(BUILD_DIR)
	@$(CXX) -o $(TARGET_GPU) $(SRC_GPU) $(CXXFLAGS) $(CXXDEFS) $(LIBS_GPU)

run-gpu: build-gpu
	@echo "Running $(TARGET_GPU)..."
	@./$(TARGET_GPU)	

# Target to generate outputs for both models with the same params
run-both: run run-gpu

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

# Target to compare frames from two model outputs
compare-frames:
	@echo "Comparing frames..."
	@$(VENV_PY) $(VALIDATE) $(COMPARE_ARGS)

.PHONY: all build run build-gpu run-gpu run-both setup-venv run-visualize compare-frames

all: build build-gpu