CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
DEPFLAGS := -MMD -MP
OMPFLAGS := -fopenmp
INCLUDES := -Iinclude

ARCHS ?= 70 75 80 86
GENCODE := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

BUILD := build
BIN   := bin

CPU_SRC  := $(wildcard src/core/*.cpp) $(wildcard src/cpu/*.cpp)
CUDA_SRC := $(wildcard src/gpu/*.cu)

CPU_OBJ  := $(CPU_SRC:%.cpp=$(BUILD)/%.o)
CUDA_OBJ := $(CUDA_SRC:%.cu=$(BUILD)/%.o)

# An nvcc only accepts host compilers up to some gcc version, and looks like a toolkit 
# older than the system glibc fails on <stdlib.h>
# NO_CUDA=1 skips the probe and builds CPU only
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

BINARIES := $(BIN)/bench $(BIN)/gen_graph $(BIN)/test_sssp
DEPS := $(OBJ:.o=.d) $(BUILD)/src/apps/bench.d $(BUILD)/src/apps/gen_graph.d \
        $(BUILD)/tests/test_sssp.d

.PHONY: all
all: $(BINARIES)

# Make compares timestamps, not flags, so switching between `make` and
# `make cpu` would otherwise keep objects built the other way. Every object
# depends on a stamp holding the flags it was built with.
CONFIG := $(BUILD)/.config
.PHONY: force
$(CONFIG): force
	@mkdir -p $(dir $@)
	@echo '$(CXXFLAGS) $(NVCCFLAGS) $(GENCODE)' | cmp -s - $@ || \
	  echo '$(CXXFLAGS) $(NVCCFLAGS) $(GENCODE)' > $@

$(BUILD)/%.o: %.cpp $(CONFIG)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/%.o: %.cu $(CONFIG)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BIN)/bench: $(BUILD)/src/apps/bench.o $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN)/gen_graph: $(BUILD)/src/apps/gen_graph.o $(BUILD)/src/core/graph.o
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN)/test_sssp: $(BUILD)/tests/test_sssp.o $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

.PHONY: test
test: $(BIN)/test_sssp
	./$(BIN)/test_sssp

.PHONY: cpu
cpu:
	$(MAKE) NO_CUDA=1 all

.PHONY: smoke
smoke: all
	@mkdir -p graphs
	./$(BIN)/gen_graph --n 20000 --m 200000 --topo uniform --out graphs/smoke.txt
	./$(BIN)/bench graphs/smoke.txt --reps 1
	./$(BIN)/test_sssp

.PHONY: sweep
sweep: all
	@tools/sweep.sh

# neeeeeed for debugging
.PHONY: info
info:
	@echo "nvcc:    $(if $(NVCC_PICK),$(NVCC_PICK),none usable, building cpu-only)"
	@echo "version: $(if $(NVCC_PICK),$(shell $(NVCC) --version | tail -1),-)"
	@echo "host cc: $(shell $(CXX) --version | head -1)"
	@echo "archs:   $(ARCHS)"
	@echo "threads: $(shell nproc)"

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(BUILD) $(BIN)

.PHONY: distclean
distclean: clean
	rm -rf graphs results
