#include "search.cuh"
#include <iostream>
#include <algorithm>
#include <limits>
#include <climits>
#include <cuda_runtime.h>
#include "../cpu/search.hpp"

#include "../common/utils.hpp"

using namespace std;

// CUDA kernel for Dijkstra's algorithm (Bellman-Ford style iteration)
__global__ void dijkstra_kernel(int* __restrict__ distances,
                                 const int* __restrict__ edges,
                                 const int* __restrict__ weights,
                                 const int* __restrict__ vertices,
                                 int num_vertices, int* changed) {
    // grid-stride loop over vertices for better scaling
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices; idx += stride) {
        int u = idx;
        int du = distances[u];
        if (du == INF) continue;

        int start = vertices[u];
        int end = vertices[u + 1];
        for (int i = start; i < end; ++i) {
            int v = edges[i];
            int w = weights[i];
            int newd = du + w;
            int old = atomicMin(&distances[v], newd);
            if (newd < old) {
                // mark that progress happened
                atomicExch(changed, 1);
            }
        }
    }
}

// CUDA kernel for BFS-based SSSP (unweighted or uniform weight)
__global__ void bfs_kernel(int* distances,
                           int* queue,
                           int* queue_size,
                           const int* __restrict__ edges,
                           const int* __restrict__ vertices,
                           int num_vertices,
                           int current_distance) {
    // snapshot of current frontier size for consistent processing
    int qsz = *queue_size;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < qsz; idx += stride) {
        int u = queue[idx];
        int start = vertices[u];
        int end = vertices[u + 1];
        for (int i = start; i < end; ++i) {
            int v = edges[i];
            if (atomicCAS(&distances[v], INF, current_distance + 1) == INF) {
                int pos = atomicAdd(queue_size, 1);
                queue[pos] = v;
            }
        }
    }
}

// GPU Dijkstra implementation
vector<int> gpu_dijkstra(const vector<int>& row_ptr,
                               const vector<int>& col_idx,
                               const vector<int>& weights,
                               int source, int num_vertices) {
    vector<int> distances(num_vertices, INF);
    distances[source] = 0;

    // CSR arrays provided: row_ptr size = num_vertices+1, col_idx and weights size = m
    int num_edges = (int)col_idx.size();

    // Allocate GPU memory
    int* d_distances; int* d_edges; int* d_weights; int* d_vertices; int* d_changed;
    cudaMalloc(&d_distances, num_vertices * sizeof(int));
    cudaMalloc(&d_edges, num_edges * sizeof(int));
    cudaMalloc(&d_weights, num_edges * sizeof(int));
    cudaMalloc(&d_vertices, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_distances, distances.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, col_idx.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices, row_ptr.data(), (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Run relaxation iterations
    int threads_per_block = 256;
    int blocks = (num_vertices + threads_per_block - 1) / threads_per_block;

    for (int iter = 0; iter < num_vertices - 1; iter++) {
        int changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);

        dijkstra_kernel<<<blocks, threads_per_block>>>(d_distances, d_edges, d_weights,
                                                        d_vertices, num_vertices, d_changed);

        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (!changed) break;
    }

    // Copy result back
    cudaMemcpy(distances.data(), d_distances, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_distances);
    cudaFree(d_edges);
    cudaFree(d_weights);
    cudaFree(d_vertices);
    cudaFree(d_changed);

    return distances;
}

// GPU Bellman-Ford implementation
vector<int> gpu_bellman_ford(const vector<int>& row_ptr,
                                   const vector<int>& col_idx,
                                   const vector<int>& weights,
                                   int source, int num_vertices) {
    // Similar implementation to Dijkstra but with full relaxation
    return gpu_dijkstra(row_ptr, col_idx, weights, source, num_vertices);
}

// GPU BFS-based SSSP for unweighted graphs
vector<int> gpu_bfs_sssp(const vector<int>& row_ptr,
                               const vector<int>& col_idx,
                               const vector<int>& weights,
                               int source, int num_vertices) {
    vector<int> distances(num_vertices, INF);
    distances[source] = 0;

    int num_edges = (int)col_idx.size();

    // Allocate GPU memory
    int* d_distances; int* d_edges; int* d_vertices; int* d_queue; int* d_queue_size;
    cudaMalloc(&d_distances, num_vertices * sizeof(int));
    cudaMalloc(&d_edges, num_edges * sizeof(int));
    cudaMalloc(&d_vertices, (num_vertices + 1) * sizeof(int));
    cudaMalloc(&d_queue, num_vertices * sizeof(int));
    cudaMalloc(&d_queue_size, sizeof(int));

    // Copy data to GPU
    cudaMemcpy(d_distances, distances.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges, col_idx.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertices, row_ptr.data(), (num_vertices + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize queue with source
    vector<int> queue(num_vertices);
    queue[0] = source;
    int queue_size = 1;

    int threads_per_block = 256;

    // BFS iterations (use frontier queue on host; after kernel read next-frontier from device)
    for (int dist = 0; dist < num_vertices; dist++) {
        if (queue_size == 0) break;

        int old_size = queue_size;
        // copy only current frontier to device
        cudaMemcpy(d_queue, queue.data(), old_size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_queue_size, &old_size, sizeof(int), cudaMemcpyHostToDevice);

        int blocks = (old_size + threads_per_block - 1) / threads_per_block;
        bfs_kernel<<<blocks, threads_per_block>>>(d_distances, d_queue, d_queue_size,
                                                   d_edges, d_vertices, num_vertices, dist);

        // get updated distances and new queue size (old + next)
        cudaMemcpy(distances.data(), d_distances, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&queue_size, d_queue_size, sizeof(int), cudaMemcpyDeviceToHost);

        int new_size = queue_size;
        int next_size = new_size - old_size;
        if (next_size <= 0) {
            queue_size = 0;
            break;
        }

        // copy only the appended next-frontier segment from device into host queue[0..next_size-1]
        cudaMemcpy(queue.data(), d_queue + old_size, next_size * sizeof(int), cudaMemcpyDeviceToHost);
        queue_size = next_size;
    }

    // Free GPU memory
    cudaFree(d_distances);
    cudaFree(d_edges);
    cudaFree(d_vertices);
    cudaFree(d_queue);
    cudaFree(d_queue_size);

    return distances;
}

void print_gpu_results(const vector<int>& distances, const string& algorithm_name) {
    cout << algorithm_name << " (GPU)" << std::endl;
    print_distances(distances);
}
