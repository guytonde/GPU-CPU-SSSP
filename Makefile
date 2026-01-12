# ====== Config ======
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native
OMPFLAGS := -fopenmp

# Directories
SRC_DIR    := src
INCLUDE_DIR:= include
TOOLS_DIR  := tools
BIN_DIR    := bin
BUILD_DIR  := build

# Source files
CPU_SERIAL_SRC    := $(SRC_DIR)/cpu_serial.cpp
CPU_PARALLEL_SRC  := $(SRC_DIR)/cpu_parallel.cpp
MAIN_SRC          := $(SRC_DIR)/main.cpp
GEN_GRAPH_SRC     := $(TOOLS_DIR)/generate_graphs.cpp

# Objects
CPU_SERIAL_OBJ    := $(BUILD_DIR)/cpu_serial.o
CPU_PARALLEL_OBJ  := $(BUILD_DIR)/cpu_parallel.o
MAIN_OBJ          := $(BUILD_DIR)/main.o
GEN_GRAPH_OBJ     := $(BUILD_DIR)/generate_graphs.o

# Binaries
BENCHMARK_BIN := $(BIN_DIR)/benchmark
GEN_GRAPH_BIN := $(BIN_DIR)/gen_graph

# Default target
.PHONY: all
all: dirs $(BENCHMARK_BIN) $(GEN_GRAPH_BIN)

# Create dirs
.PHONY: dirs
dirs:
	@mkdir -p $(BIN_DIR) $(BUILD_DIR)

# ====== Compilation rules ======
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/main.o: $(MAIN_SRC) | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

$(BUILD_DIR)/generate_graphs.o: $(GEN_GRAPH_SRC) | dirs
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# ====== Link binaries ======
$(BENCHMARK_BIN): $(MAIN_OBJ) $(CPU_SERIAL_OBJ) $(CPU_PARALLEL_OBJ) | dirs
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $^ -I$(INCLUDE_DIR) -o $@
	@echo "Built $@"

$(GEN_GRAPH_BIN): $(GEN_GRAPH_OBJ) | dirs
	$(CXX) $(CXXFLAGS) $^ -o $@
	@echo "Built $@"

# ====== Convenience targets ======

.PHONY: benchmark gen_graph
benchmark: $(BENCHMARK_BIN)
gen_graph: $(GEN_GRAPH_BIN)

.PHONY: quicktest
quicktest: all
	@$(GEN_GRAPH_BIN) 100 500 1 10 graphs/generated/tiny.txt
	@$(BENCHMARK_BIN) graphs/generated/tiny.txt 0

.PHONY: bench
bench: all
	@chmod +x $(TOOLS_DIR)/run_benchmarks.sh
	@$(TOOLS_DIR)/run_benchmarks.sh

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: distclean
distclean: clean
	rm -rf graphs/generated results
