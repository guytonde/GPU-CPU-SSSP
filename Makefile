CXX      ?= g++
NVCC     ?= nvcc
CXXFLAGS ?= -std=c++17 -O3 -march=native -Wall -Wextra
NVCCFLAGS ?= -std=c++17 -O3 -lineinfo
DEPFLAGS := -MMD -MP
OMPFLAGS := -fopenmp
INCLUDES := -Iinclude
PYTHON   ?= python3

# The installed card's compute capability, or a spread of common ones.
ARCHS ?= $(or $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .),70 75 80 86 89)
GENCODE := $(foreach a,$(ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

BUILD := build
BIN   := bin

CPU_SRC  := $(wildcard src/core/*.cpp) $(wildcard src/cpu/*.cpp)
CUDA_SRC := $(wildcard src/gpu/*.cu)

CPU_OBJ  := $(CPU_SRC:%.cpp=$(BUILD)/%.o)
CUDA_OBJ := $(CUDA_SRC:%.cu=$(BUILD)/%.o)

# Use the first nvcc that compiles a trivial file with the host compiler.
# NO_CUDA=1 builds CPU only.
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
  CUDA_HOME := $(abspath $(dir $(shell command -v $(NVCC)))/..)
  CXXFLAGS += -DSSSP_CUDA
  NVCCFLAGS += -DSSSP_CUDA
  # for nvml.h; the library is loaded at run time
  INCLUDES += -isystem $(CUDA_HOME)/include
  LDFLAGS += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart
  OBJ := $(CPU_OBJ) $(CUDA_OBJ)
else
  OBJ := $(CPU_OBJ)
endif
LDFLAGS += -ldl

BINARIES := $(BIN)/bench $(BIN)/gen_graph $(BIN)/calibrate $(BIN)/test_sssp
APPS := bench gen_graph calibrate
DEPS := $(OBJ:.o=.d) $(APPS:%=$(BUILD)/src/apps/%.d) $(BUILD)/tests/test_sssp.d

.PHONY: all
all: $(BINARIES)

# Keep object files, so make does not relink the apps every time.
.SECONDARY:

# Changes when the flags change, so objects rebuild.
CONFIG := $(BUILD)/.config
.PHONY: force
$(CONFIG): force
	@mkdir -p $(dir $@)
	@echo '$(CXXFLAGS) $(NVCCFLAGS) $(GENCODE)' | cmp -s - $@ || \
	  echo '$(CXXFLAGS) $(NVCCFLAGS) $(GENCODE)' > $@

# Every run records the commit, whether the tree was dirty, and a hash of the
# uncommitted changes to the code.
CODE_PATHS := src include Makefile
GIT_SHA   := $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell test -z "$$(git status --porcelain -- $(CODE_PATHS) 2>/dev/null)" && echo 0 || echo 1)
DIFF_HASH := $(shell (git diff HEAD -- $(CODE_PATHS); git ls-files --others --exclude-standard -- $(CODE_PATHS) | sort | xargs -r cat) 2>/dev/null | sha1sum | cut -c1-12)
BUILD_DESC := $(subst ",,$(CXX) $(CXXFLAGS) | $(NVCCFLAGS) | archs $(ARCHS))
GITSTAMP := $(BUILD)/.gitstamp
$(GITSTAMP): force
	@mkdir -p $(dir $@)
	@echo '$(GIT_SHA) $(GIT_DIRTY) $(DIFF_HASH)' | cmp -s - $@ || \
	  echo '$(GIT_SHA) $(GIT_DIRTY) $(DIFF_HASH)' > $@

$(BUILD)/src/core/build_info.o: $(GITSTAMP)
$(BUILD)/src/core/build_info.o: CXXFLAGS += -DSSSP_GIT_SHA='"$(GIT_SHA)"' \
  -DSSSP_GIT_DIRTY=$(GIT_DIRTY) -DSSSP_DIFF_HASH='"$(DIFF_HASH)"' \
  -DSSSP_BUILD_FLAGS='"$(BUILD_DESC)"'

$(BUILD)/%.o: %.cpp $(CONFIG)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(DEPFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD)/%.o: %.cu $(CONFIG)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(GENCODE) $(DEPFLAGS) -Iinclude -c $< -o $@

$(BIN)/%: $(BUILD)/src/apps/%.o $(OBJ)
	@mkdir -p $(BIN)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -o $@ $(LDFLAGS)

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
	./$(BIN)/test_sssp
	$(PYTHON) tools/experiments.py --quick smoke

# Everything except experiment F, which needs sudo (tools/profile.sh).
.PHONY: reproduce
reproduce: all
	./$(BIN)/test_sssp
	$(PYTHON) tools/experiments.py all
	$(PYTHON) tools/analyze.py
	$(PYTHON) tools/plot.py

.PHONY: analyze
analyze:
	$(PYTHON) tools/analyze.py

.PHONY: figures
figures:
	$(PYTHON) tools/plot.py

.PHONY: info
info:
	@echo "nvcc:    $(if $(NVCC_PICK),$(NVCC_PICK),none usable, building cpu-only)"
	@echo "version: $(if $(NVCC_PICK),$(shell $(NVCC) --version | tail -1),-)"
	@echo "host cc: $(shell $(CXX) --version | head -1)"
	@echo "archs:   $(ARCHS)"
	@echo "git:     $(GIT_SHA) dirty=$(GIT_DIRTY) diff=$(DIFF_HASH)"

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(BUILD) $(BIN)

.PHONY: distclean
distclean: clean
	rm -rf graphs
