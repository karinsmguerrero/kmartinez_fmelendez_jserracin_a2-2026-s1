#General makefile for the project
CXX ?= g++
CXXFLAGS = -Wall -Wextra -std=c++11 -fno-fast-math -fexcess-precision=standard

# Extra flags for SIMD sources (enable AVX2 and FMA, optimize)
CXXFLAGS_SIMD = $(CXXFLAGS) -mavx2 -mfma -O3

PYTHON ?= python3
VISUALIZE = scripts/visualize.py
VALIDATE = scripts/validation.py
VENV ?= .venv
VENV_PY = $(VENV)/bin/python
REQS = requirements.txt
COMPARE_ARGS ?=
NBODIES ?= 14000
NSTEPS ?= 100
NBSAVEFRAMES ?= 1
NBVERBOSE ?= 0
RUNS ?= 60

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

RESULTS_DIR = results
PERF_CSV = $(RESULTS_DIR)/perf_results.csv
PERF_GPU_CSV = $(RESULTS_DIR)/perf_results_$(NBODIES).csv

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


SRC_SIMD = models/cpu_simd.cpp
TARGET_SIMD = $(BUILD_DIR)/cpu_simd

build-simd: $(SRC_SIMD)
	@echo "Compiling $(SRC_SIMD)..."
	@mkdir -p $(BUILD_DIR)
	@$(CXX) -o $(TARGET_SIMD) $(SRC_SIMD) $(CXXFLAGS_SIMD) $(CXXDEFS)

run-simd: build-simd
	@echo "Running $(TARGET_SIMD)..."
	@./$(TARGET_SIMD)

# Run perf measurements for sequential, SIMD and GPU builds
.PHONY: perf
perf:
	@echo "Running perf measurements ($(RUNS) runs each)..."
	@mkdir -p $(RESULTS_DIR)
	@echo "nbodies,run,program,time_sec" > $(PERF_CSV)

	@for n in 10000 12000 14000 16000 18000 20000; do \
		echo ""; \
		echo "========================================"; \
		echo "NBODIES=$$n"; \
		echo "========================================"; \
		\
		$(MAKE) build NBODIES=$$n; \
		$(MAKE) build-simd NBODIES=$$n; \
		$(MAKE) build-gpu NBODIES=$$n; \
		\
		for prog in $(TARGET) $(TARGET_SIMD) $(TARGET_GPU); do \
			for i in $(shell seq 1 $(RUNS)); do \
				echo "Measuring $$prog (NBODIES=$$n, run $$i)..."; \
				out=$$( (LANG=C perf stat -x, -e duration_time -- "$$prog") 2>&1 1>/dev/null ); \
				dur_ns=$$(echo "$$out" | awk -F, '$$3=="duration_time" {print $$1; exit}'); \
				time_sec=$$(awk 'BEGIN{ns='"$$dur_ns"'; if (ns=="" || ns=="<not counted>") {print ""} else {printf "%.9f", ns/1e9}}'); \
				echo "$$n,$$i,$$(basename $$prog),$${time_sec}" >> $(PERF_CSV); \
			done; \
		done; \
	done

	@echo "Results written to $(PERF_CSV)"

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

run-profile: build-gpu
	@rocprof-sys-sample --gpus 0 --G L2CacheHit GL2C_MISS GL2C_MISS_sum GPUBusy GRBM_COUNT \
	LdsLatency MeanOccupancyPerActiveCU MeanOccupancyPerCU OccupancyPercent SQ_WAVE_CYCLES \
	Wavefronts --realtime 0 --output "/home/karina/Documentos/TEC/Arqui_2/Proyectos/Proyecto 3/proyecto_2/profiling/results" n_body_gpu_$(NBODIES) -- build/n-body_gpu
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

.PHONY: all build run run-simd build-gpu run-gpu run-both setup-venv run-visualize compare-frames

all: build build-gpu build-simd
