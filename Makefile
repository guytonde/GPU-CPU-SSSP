CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
OMPFLAGS := -fopenmp
INCLUDES := -Iinclude

# Compute capabilities to embed. sm_70 covers Volta through Ampere by PTX JIT;
# add or trim to match the machine you actually benchmark on.
ARCHS ?= 70 75 80 86
GENCODE := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

BUILD := build
BIN   := bin

CPU_SRC  := $(wildcard src/core/*.cpp) $(wildcard src/cpu/*.cpp)
CUDA_SRC := $(wildcard src/gpu/*.cu)

CPU_OBJ  := $(CPU_SRC:%.cpp=$(BUILD)/%.o)
CUDA_OBJ := $(CUDA_SRC:%.cu=$(BUILD)/%.o)

# An nvcc only accepts host compilers up to some gcc version, and a toolkit
# older than the system glibc fails on <stdlib.h> long before it reaches our
# code. So instead of trusting whatever is on PATH, compile a trivial .cu with
# each candidate and take the first that survives. Set NO_CUDA=1 to skip the
# probe and build CPU-only; the gpu solvers just drop out of the registry.
CUDA_SEARCH := $(NVCC) $(wildcard /usr/local/cuda*/bin/nvcc /lusr/opt/cuda-*/bin/nvcc)

ifneq ($(filter clean distclean,$(MAKECMDGOALS)),)
  NO_CUDA := 1
endif

ifndef NO_CUDA
NVCC_PICK := $(shell d=$$(mktemp -d); \
  printf '#include <cuda_runtime.h>\nint main(){return 0;}\n' > $$d/p.cu; \
  for c in $(CUDA_SEARCH); do \
    command -v $$c >/dev/null 2>&1 || continue; \
    $$c -std=c++17 -c $$d/p.cu -o $$d/p.o >/dev/null 2>&1 && { echo $$c; break; }; \
  done; rm -rf $$d)
endif

ifneq ($(NVCC_PICK),)
  NVCC := $(NVCC_PICK)
  CXXFLAGS += -DSSSP_CUDA
  NVCCFLAGS += -DSSSP_CUDA
  CUDA_LIBDIR := $(abspath $(dir $(shell command -v $(NVCC)))/../lib64)
  LDFLAGS += -L$(CUDA_LIBDIR) -Wl,-rpath,$(CUDA_LIBDIR) -lcudart
  OBJ := $(CPU_OBJ) $(CUDA_OBJ)
else
  OBJ := $(CPU_OBJ)
endif

BINARIES := $(BIN)/bench $(BIN)/gen_graph

.PHONY: all
all: $(BINARIES)

$(BUILD)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/%.o: %.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(INCLUDES) -c $< -o $@

$(BIN)/bench: $(BUILD)/src/apps/bench.o $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/gen_graph: $(BUILD)/src/apps/gen_graph.o $(BUILD)/src/core/graph.o
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

.PHONY: cpu
cpu:
	$(MAKE) NO_CUDA=1 all

.PHONY: smoke
smoke: all
	@mkdir -p graphs
	./$(BIN)/gen_graph --n 20000 --m 200000 --topo uniform --out graphs/smoke.txt
	./$(BIN)/bench graphs/smoke.txt --reps 1

.PHONY: sweep
sweep: all
	@tools/sweep.sh

.PHONY: info
info:
	@echo "nvcc:    $(if $(NVCC_PICK),$(NVCC_PICK),none usable, building cpu-only)"
	@echo "version: $(if $(NVCC_PICK),$(shell $(NVCC) --version | tail -1),-)"
	@echo "host cc: $(shell $(CXX) --version | head -1)"
	@echo "archs:   $(ARCHS)"
	@echo "threads: $(shell nproc)"

.PHONY: clean
clean:
	rm -rf $(BUILD) $(BIN)

.PHONY: distclean
distclean: clean
	rm -rf graphs results
