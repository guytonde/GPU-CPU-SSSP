CXX = g++
NVCC = nvcc
HARDENING_FLAGS = -D_FORTIFY_SOURCE=2 -fstack-protector-strong -fstack-clash-protection -fcf-protection=full -fPIE
CXXFLAGS = -O2 -std=c++17 -Wall $(HARDENING_FLAGS)
NVCCFLAGS = -O2 -std=c++17 -arch=sm_70 -DUSE_GPU

# Targets
GEN_GRAPH = bin/generate_graph
SSSP_CPU = bin/sssp_cpu
SSSP_GPU = bin/sssp_gpu

# Source files
GEN_GRAPH_SRC = graphs/generate_graphs.cpp
CPU_SRCS = src/cpu/search.cpp src/main.cpp
GPU_SRCS = src/gpu/search.cu src/cpu/search.cpp src/main.cpp

# Build directory
BUILD_DIR = build
BIN_DIR = bin

# Create directories
$(shell mkdir -p $(BUILD_DIR) $(BIN_DIR))

.PHONY: all clean graph cpu gpu test help test-cpu test-gpu test-all

all: cpu gpu

help:
	@echo "Targets:"
	@echo "  make cpu                 - Build CPU version"
	@echo "  make gpu                 - Build GPU version (requires CUDA)"
	@echo "  make all                 - Build both CPU and GPU versions"
	@echo "  make graph V=N E=M ...   - Generate a test graph"
	@echo "  make test-cpu            - Build and test CPU version"
	@echo "  make test-gpu            - Build and test GPU version"
	@echo "  make test-all            - Build and test both CPU and GPU versions"
	@echo "  make clean               - Remove all build artifacts"
	@echo ""
	@echo "Graph generation:"
	@echo "  make graph V=1000 E=5000 MINW=1 MAXW=100 OUT=graph.txt"
	@echo ""
	@echo "Run examples:"
	@echo "  ./bin/sssp_cpu dijkstra graph.txt 1000 0"
	@echo "  ./bin/sssp_cpu bellman-ford graph.txt 1000 0"
	@echo "  ./bin/sssp_cpu bfs graph.txt 1000 0"
	@echo "  ./bin/sssp_cpu johnson graph.txt 1000"
	@echo "  ./bin/sssp_gpu gpu-dijkstra graph.txt 1000 0"

# Generate graph generator
$(GEN_GRAPH): $(GEN_GRAPH_SRC)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build CPU version
cpu: $(SSSP_CPU)

$(SSSP_CPU): $(CPU_SRCS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Build GPU version
gpu: $(SSSP_GPU)

$(SSSP_GPU): $(GPU_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Generate a test graph
graph: $(GEN_GRAPH)
	@if [ -z "$(V)" ] || [ -z "$(E)" ] || [ -z "$(MINW)" ] || [ -z "$(MAXW)" ] || [ -z "$(OUT)" ]; then \
		echo "Usage: make graph V=<vertices> E=<edges> MINW=<min_weight> MAXW=<max_weight> OUT=<output_file>"; \
		exit 1; \
	fi
	./$(GEN_GRAPH) $(V) $(E) $(MINW) $(MAXW) $(OUT)
	@echo "Graph saved to $(OUT)"

# Generate a default test graph for testing
graphs/generated/test_graph.txt: $(GEN_GRAPH)
	@mkdir -p graphs/generated
	./$(GEN_GRAPH) 100 500 1 50 graphs/generated/test_graph.txt

# Test CPU algorithms
test-cpu: cpu graphs/generated/test_graph.txt
	@echo "Testing CPU Dijkstra..."
	./$(SSSP_CPU) dijkstra graphs/generated/test_graph.txt 100 0
	@echo ""
	@echo "Testing CPU Bellman-Ford..."
	./$(SSSP_CPU) bellman-ford graphs/generated/test_graph.txt 100 0
	@echo ""
	@echo "Testing CPU BFS..."
	./$(SSSP_CPU) bfs graphs/generated/test_graph.txt 100 0

# Test GPU algorithms
test-gpu: gpu graphs/generated/test_graph.txt
	@echo "Testing GPU Dijkstra..."
	./$(SSSP_GPU) gpu-dijkstra graphs/generated/test_graph.txt 100 0
	@echo ""
	@echo "Testing GPU Bellman-Ford..."
	./$(SSSP_GPU) gpu-bellman-ford graphs/generated/test_graph.txt 100 0
	@echo ""
	@echo "Testing GPU BFS..."
	./$(SSSP_GPU) gpu-bfs graphs/generated/test_graph.txt 100 0

# Test all algorithms
test-all: cpu gpu graphs/generated/test_graph.txt
	@echo "Running tests and generating summary table..."
	@printf "%-20s | %-12s | %-12s | %-8s\n" "Algorithm" "CPU Time(us)" "GPU Time(us)" "Diffs"
	@printf "%-20s-+-%-12s-+-%-12s-+-%-8s\n" "--------------------" "------------" "------------" "--------"
	@tools/test_summary.sh

clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) graphs/generated/
	find . -name "*.o" -delete
	find . -name "*.a" -delete
