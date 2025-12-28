# GPU-CPU-SSSP
An investigation in CPU, GPU, and hybrid implementations for Single-Source Shortest Path (SSSP) algorithms. This project tests the performance benefits of various implementations of search algorithms on different hardware.

## Overview

This project implements several SSSP algorithms on both CPU and GPU:

### Algorithms
- **Dijkstra's Algorithm** - For non-negative edge weights
- **Bellman-Ford Algorithm** - For graphs with negative edge weights
- **BFS-based SSSP** - For unweighted graphs
- **Johnson's Algorithm** - For all-pairs shortest paths on CPU
- **GPU Implementations** - Parallel versions of Dijkstra and Bellman-Ford using CUDA

## Requirements

### CPU Build
- G++ compiler 
- GNU Make

### GPU Build
- NVIDIA CUDA Toolkit (I have 12.2) and compatible GPU
- NVCC compiler

## Building

### Build CPU Version Only
```bash
make cpu
```

### Build GPU Version (requires CUDA)
```bash
make gpu
```

### View All Build Options
```bash
make help
```

### Clean Build Artifacts
```bash
make clean
```

## Generating Test Graphs

Before running the algorithms, generate a test graph:

```bash
make graph V=1000 E=5000 MINW=1 MAXW=100 OUT=my_graph.txt
```

Parameters:
- `V` - Number of vertices
- `E` - Number of edges (must be ≤ V*(V-1)/2)
- `MINW` - Minimum edge weight
- `MAXW` - Maximum edge weight
- `OUT` - Output filename

## Running

### CPU Algorithms

**Dijkstra's Algorithm:**
```bash
./bin/sssp_cpu dijkstra graph.txt <num_vertices> <source_vertex>
```

**Bellman-Ford Algorithm:**
```bash
./bin/sssp_cpu bellman-ford graph.txt <num_vertices> <source_vertex>
```

**BFS-based SSSP (for unweighted graphs):**
```bash
./bin/sssp_cpu bfs graph.txt <num_vertices> <source_vertex>
```

**Johnson's Algorithm (all-pairs):**
```bash
./bin/sssp_cpu johnson graph.txt <num_vertices>
```

### GPU Algorithms

**GPU Dijkstra:**
```bash
./bin/sssp_gpu gpu-dijkstra graph.txt <num_vertices> <source_vertex>
```

**GPU Bellman-Ford:**
```bash
./bin/sssp_gpu gpu-bellman-ford graph.txt <num_vertices> <source_vertex>
```

**GPU BFS:**
```bash
./bin/sssp_gpu gpu-bfs graph.txt <num_vertices> <source_vertex>
```

## Using the makefile

```bash
# Build the CPU version
make cpu

# Generate a test graph with 5000 vertices and 20000 edges
make graph V=5000 E=20000 MINW=1 MAXW=100 OUT=test_graph.txt

# Run Dijkstra's algorithm from vertex 0
./bin/sssp_cpu dijkstra test_graph.txt 5000 0

# For GPU (if CUDA is available)
make gpu
./bin/sssp_gpu gpu-dijkstra test_graph.txt 5000 0
```

## Graph File Format

The graph file format is simple (one edge per line):
```
<source> <destination> <weight>
```

## Work as of now
Currently doing testing. I'm getting experiemnt data like this and think my kernels are just inefficient in many ways. Going ot revisit that as I'm still a bit unfamiliar. You can run this using the tools. I made benchmark_3d.sh to vary the 3 dimensions of # vert, # edges, and magnitude of weight. Not testing a* for now.

```
Summary (sorted by V, E, weight range):
     V      E     MinW     MaxW          Algorithm      CPU(us)      GPU(us)  Diffs
------ ------ -------- -------- ------------------   ----------   ---------- ------
   100    200        1     1000       bellman-ford           20        32051      0
   100    200        1     1000                bfs            2        48499      0
   100    200        1     1000           dijkstra            5        33940      0
   100    200        1      100       bellman-ford           23        31963      0
   100    200        1      100                bfs            2        31594      0
   100    200        1      100           dijkstra            6        48350      0
   100    200        1       10       bellman-ford           20        60607      0
   100    200        1       10                bfs            1        31367      0
   100    200        1       10           dijkstra           11        76096      0
   100    500        1     1000       bellman-ford           32        50127      0
   100    500        1     1000                bfs            2        34549      0
   100    500        1     1000           dijkstra            9        33343      0
   100    500        1      100       bellman-ford           31        31806      0
   100    500        1      100                bfs            2        46407      0
   100    500        1      100           dijkstra            9        32108      0
   100    500        1       10       bellman-ford           33        32731      0
   100    500        1       10                bfs            2        47676      0
   100    500        1       10           dijkstra            9        31298      0
   100   1000        1     1000       bellman-ford           50        46578      0
   100   1000        1     1000                bfs            2        32008      0
   100   1000        1     1000           dijkstra           20        47478      0
   100   1000        1      100       bellman-ford           55        32371      0
   100   1000        1      100                bfs            2        32160      0
   100   1000        1      100           dijkstra           15        33066      0
   100   1000        1       10       bellman-ford           48        42972      0
   100   1000        1       10                bfs            2        47890      0
   100   1000        1       10           dijkstra           12        46694      0
   500   1000        1     1000       bellman-ford           65        32347      0
   500   1000        1     1000                bfs            0        32128      0
   500   1000        1     1000           dijkstra            0        32801      0
   500   1000        1      100       bellman-ford           65        32180      0
   500   1000        1      100                bfs            0        32155      0
   500   1000        1      100           dijkstra            0        31706      0
   500   1000        1       10       bellman-ford           67        45354      0
   500   1000        1       10                bfs            0        31870      0
   500   1000        1       10           dijkstra            0        35987      0
   500   2500        1     1000       bellman-ford          665        31641      0
   500   2500        1     1000                bfs            9        34829      0
   500   2500        1     1000           dijkstra           49        31947      0
   500   2500        1      100       bellman-ford          897        32236      0
   500   2500        1      100                bfs            9        43682      0
   500   2500        1      100           dijkstra           77        33930      0
   500   2500        1       10       bellman-ford          663        47989      0
   500   2500        1       10                bfs           12        34066      0
   500   2500        1       10           dijkstra           50        46666      0
   500   5000        1     1000       bellman-ford         1462        32861      0
   500   5000        1     1000                bfs           10        46448      0
   500   5000        1     1000           dijkstra           71        49780      0
   500   5000        1      100       bellman-ford         1434        32889      0
   500   5000        1      100                bfs           12        33708      0
   500   5000        1      100           dijkstra          161        32533      0
   500   5000        1       10       bellman-ford         1290        34677      0
   500   5000        1       10                bfs           10        47170      0
   500   5000        1       10           dijkstra           68        43735      0
  1000   2000        1     1000       bellman-ford         1602        32583      0
  1000   2000        1     1000                bfs           13        32553      0
  1000   2000        1     1000           dijkstra           63        32575      0
  1000   2000        1      100       bellman-ford         1576        45954      0
  1000   2000        1      100                bfs           13        45264      0
  1000   2000        1      100           dijkstra           55        32978      0
  1000   2000        1       10       bellman-ford         1578        33953      0
  1000   2000        1       10                bfs           12        32613      0
  1000   2000        1       10           dijkstra           56        47263      0
  1000   5000        1     1000       bellman-ford         2806        32972      0
  1000   5000        1     1000                bfs           20        47984      0
  1000   5000        1     1000           dijkstra          104        31781      0
  1000   5000        1      100       bellman-ford         2899        32094      0
  1000   5000        1      100                bfs           20        44898      0
  1000   5000        1      100           dijkstra          114        47737      0
  1000   5000        1       10       bellman-ford         2945        31866      0
  1000   5000        1       10                bfs           20        45205      0
  1000   5000        1       10           dijkstra          107        33483      0
  1000  10000        1     1000       bellman-ford         7431        46134      0
  1000  10000        1     1000                bfs           22        32351      0
  1000  10000        1     1000           dijkstra          240        35710      0
  1000  10000        1      100       bellman-ford         7387        31708      0
  1000  10000        1      100                bfs           29        47645      0
  1000  10000        1      100           dijkstra          258        34032      0
  1000  10000        1       10       bellman-ford         7508        32146      0
  1000  10000        1       10                bfs           22        32575      0
  1000  10000        1       10           dijkstra          134        32408      0
  ```