# Summary

13th Gen Intel(R) Core(TM) i9-13900KF, NVIDIA GeForce RTX 4090, driver 575.57.08. 27246 timed runs over 62 graphs, plus 803 profiling and placement runs.

## Calibration constants

| constant | value |
|---|---|
| alpha_frontier_us | 14.421 |
| theta_uniform_deg8_edges_per_ns | 4.476 |
| cost_per_vertex_uniform_deg8_ns | 1.787 |
| theta_grid_deg4_edges_per_ns | 1.996 |
| cost_per_vertex_grid_deg4_ns | 2.003 |
| lambda_gpu-topo_ns | 133.038 |
| h2d_pageable_1MB_GBps | 12.998 |
| h2d_pageable_16MB_GBps | 16.821 |
| h2d_pageable_1024MB_GBps | 16.812 |
| h2d_pinned_1MB_GBps | 17.985 |
| h2d_pinned_16MB_GBps | 20.162 |
| h2d_pinned_1024MB_GBps | 20.306 |
| omp_region_24_threads_us | 2.089 |

## Model (fitted on A, median error elsewhere)

| solver | model | alpha_us | ns_per_edge | ns_per_vertex | error_A | error_B | error_C | error_E | error_HO |
|---|---|---|---|---|---|---|---|---|---|
| gpu-frontier | syncs + edges | 15.968 | 0.064 |  | 0.259 | 0.450 | 0.431 | 0.416 | 0.414 |
| gpu-frontier | syncs + vertices + edges | 13.393 | 0.015 | 0.844 | 0.029 | 0.370 | 0.048 | 0.129 | 0.062 |
| gpu-nearfar | syncs + edges | 5.622 | 0.120 |  | 0.069 |  | 0.080 | 0.080 | 0.081 |
| gpu-nearfar | syncs + vertices + edges | 5.655 | 0.026 | 0.781 | 0.061 |  | 0.068 | 0.072 | 0.072 |
| gpu-frontier | calibration | 14.421 | 0.223 | 1.787 |  |  |  |  |  |

## Picking the faster device

| rule | accuracy | regret | cases |
|---|---|---|---|
| always CPU | 0.365 | 0.945 | 260 |
| always GPU | 0.635 | 0.245 | 260 |
| BFS frontier width, fitted | 0.862 | 0.079 | 260 |
| graph properties and calibrated constants | 0.877 | 0.060 | 260 |
| model on the run's own counters, true CPU time | 0.992 | 0.001 | 260 |

## Break even query counts

| label | gpu_solver | best_cpu | q_cpu_ms | q_gpu_ms | U_ms | Kstar_measured | Kstar_predicted | q_batch_ms | Kstar_vs_batch |
|---|---|---|---|---|---|---|---|---|---|
| layered W=256 u[1,100] | gpu-frontier | dial | 97.72 | 166.12 | 13.84 | never | never | 8.62 | never |
| layered W=256 u[1,100] | gpu-nearfar | dial | 97.72 | 280.82 | 3.27 | never | never | 8.62 | never |
| layered W=4096 u[1,100] | gpu-frontier | delta-omp | 45.51 | 12.45 | 5.35 | 1.00 | 1.00 | 11.49 | never |
| layered W=4096 u[1,100] | gpu-nearfar | delta-omp | 45.51 | 24.32 | 4.61 | 1.00 | 1.00 | 11.49 | never |
| layered W=65536 u[1,100] | gpu-frontier | delta-omp | 18.80 | 4.13 | 5.36 | 1.00 | 1.00 | 16.79 | 1.00 |
| layered W=65536 u[1,100] | gpu-nearfar | delta-omp | 18.80 | 5.37 | 4.51 | 1.00 | 1.00 | 16.79 | 1.00 |
| uniform n=1048576 u[1,100] | gpu-frontier | delta-omp | 18.92 | 4.31 | 4.83 | 1.00 | 1.00 | 17.65 | 1.00 |
| uniform n=1048576 u[1,100] | gpu-nearfar | delta-omp | 18.92 | 5.01 | 4.90 | 1.00 | 1.00 | 17.65 | 1.00 |
| grid 1024x1024 u[1,100] | gpu-frontier | dial | 47.53 | 61.94 | 1.99 | never | never | 5.08 | never |
| grid 1024x1024 u[1,100] | gpu-nearfar | dial | 47.53 | 130.84 | -4.94 | never | never | 5.08 | never |

## Hardware counters

| kernel | launches | kernel_us | sm_active_pct | occupancy_pct | sm_tail | dram_pct | l2_pct | issue_active_pct | lanes_active | sectors_per_request | bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_frontier | 40 | 3.39 | 64.55 | 15.82 | 1.15 | 3.40 | 6.58 | 6.18 | 24.95 | 1.12 | too little parallelism |
| hub_frontier | 22 | 5.65 | 59.18 | 47.25 | 48.88 | 7.01 | 12.48 | 7.09 | 26.12 | 1.29 | load imbalance |
| hub_topo | 16 | 10668.93 | 2.82 | 60.46 | 40.08 | 0.63 | 1.23 | 4.32 | 18.56 | 12.09 | load imbalance |
| layered_frontier | 40 | 15.30 | 89.84 | 72.84 | 1.05 | 10.79 | 17.96 | 9.25 | 25.61 | 1.29 | memory latency |
| uniform_edge | 19 | 638.34 | 100.52 | 87.72 | 1.00 | 58.20 | 67.02 | 5.16 | 31.85 | 3.32 | bandwidth |
| uniform_frontier | 32 | 35.54 | 100.45 | 83.32 | 1.05 | 9.97 | 9.39 | 3.64 | 25.00 | 1.29 | memory latency |
| uniform_nearfar | 60 | 3.58 | 100.04 | 77.93 | 1.93 | 11.52 | 10.40 | 4.12 | 24.57 | 1.32 | memory latency |
| uniform_topo | 21 | 803.42 | 100.01 | 86.97 | 1.00 | 30.52 | 58.15 | 3.93 | 19.24 | 13.66 | memory latency |

Nsight Systems, grid, gpu-frontier, 4 solves: 35.3 ms in kernels, 97.4 ms in 6009 blocking copies.

| solver | graph | ms_per_solve | ipc_p_core | ipc_e_core | e_core_share | llc_misses_per_1k_instructions | thread_utilization |
|---|---|---|---|---|---|---|---|
| delta-omp | grid_unit | 48.67 | 0.78 | 0.68 | 0.37 | 2.08 | 0.04 |
| delta-omp | uniform_2e22 | 113.00 | 0.23 | 0.16 | 0.34 | 9.44 | 0.92 |
| dial | grid_unit | 40.51 | 0.56 | 0.49 | 0.39 | 3.12 | 1.00 |
| dial | uniform_2e22 | 650.92 | 0.25 | 0.54 | 0.59 | 8.48 | 1.00 |
| dijkstra | grid_unit | 108.48 | 0.68 | 0.67 | 0.42 | 1.10 | 1.00 |
| dijkstra | uniform_2e22 | 1743.52 | 0.49 | 0.46 | 0.37 | 4.40 | 1.00 |

## Serial CPU solvers pinned to one P core

| graph | solver | pcore | unpinned | unpinned_over_pinned |
|---|---|---|---|---|
| grid | delta | 78.37 | 77.95 | 0.99 |
| grid | dial | 47.31 | 47.11 | 1.00 |
| grid | dijkstra | 152.07 | 151.41 | 1.00 |
| layered | delta | 151.29 | 150.16 | 0.99 |
| layered | dial | 98.20 | 97.10 | 0.99 |
| layered | dijkstra | 281.31 | 279.72 | 0.99 |
| uniform | delta | 174.65 | 173.37 | 0.99 |
| uniform | dial | 101.46 | 100.42 | 0.99 |
| uniform | dijkstra | 310.92 | 309.74 | 1.00 |

## Per graph

| experiment | label | Pi | best_cpu | best_gpu | S1 | Sinf | Kstar | iota |
|---|---|---|---|---|---|---|---|---|
| A | layered W=1024 k=16 u[1,100] | 29638.96 | dial | gpu-frontier | 3.07 | 4.40 | 1.00 | 25.44 |
| A | layered W=1024 natural u[1,100] | 7187.33 | dial | gpu-frontier | 0.64 | 0.72 | never | 31.79 |
| A | layered W=1024 u[1,100] | 7187.33 | dial | gpu-frontier | 1.77 | 1.99 | 1.00 | 31.79 |
| A | layered W=1024 unit | 8176.02 | dial | gpu-frontier | 3.78 | 5.38 | 1.00 | 1.00 |
| A | layered W=16 u[1,100] | 117.76 | dial | gpu-frontier | 0.05 | 0.05 | never | 1386.39 |
| A | layered W=16 unit | 128.00 | dial | gpu-frontier | 0.08 | 0.09 | never | 1.00 |
| A | layered W=16384 k=16 u[1,100] | 377499.69 | delta-omp | gpu-frontier | 2.22 | 8.94 | 1.00 | 3.66 |
| A | layered W=16384 natural u[1,100] | 92781.30 | delta-omp | gpu-frontier | 1.82 | 3.84 | 1.00 | 3.71 |
| A | layered W=16384 u[1,100] | 92781.30 | delta-omp | gpu-frontier | 2.49 | 5.27 | 1.00 | 3.71 |
| A | layered W=16384 unit | 127039.02 | delta-omp | gpu-frontier | 1.41 | 5.07 | 1.00 | 1.00 |
| A | layered W=256 u[1,100] | 1827.93 | dial | gpu-frontier | 0.54 | 0.56 | never | 112.55 |
| A | layered W=256 unit | 2047.50 | dial | gpu-frontier | 1.25 | 1.38 | 1.00 | 1.00 |
| A | layered W=262144 u[1,100] | 202950.19 | delta-omp | gpu-nearfar | 1.74 | 4.12 | 1.00 | 1.04 |
| A | layered W=262144 unit | 571950.55 | delta-omp | gpu-frontier | 1.53 | 6.40 | 1.00 | 1.00 |
| A | layered W=4096 u[1,100] | 27085.45 | delta-omp | gpu-frontier | 1.81 | 2.66 | 1.00 | 8.57 |
| A | layered W=4096 unit | 32513.00 | delta-omp | gpu-frontier | 6.03 | 15.54 | 1.00 | 1.00 |
| A | layered W=64 k=16 u[1,100] | 1889.74 | dial | gpu-frontier | 0.31 | 0.32 | never | 322.30 |
| A | layered W=64 natural u[1,100] | 459.50 | dial | gpu-frontier | 0.05 | 0.05 | never | 459.69 |
| A | layered W=64 u[1,100] | 459.50 | dial | gpu-frontier | 0.14 | 0.15 | never | 459.69 |
| A | layered W=64 unit | 512.00 | dial | gpu-frontier | 0.34 | 0.34 | never | 1.00 |
| A | layered W=65536 u[1,100] | 224694.86 | delta-omp | gpu-frontier | 1.80 | 4.38 | 1.00 | 3.13 |
| A | layered W=65536 unit | 462607.06 | delta-omp | gpu-frontier | 1.50 | 6.80 | 1.00 | 1.00 |
| C | grid 1024x1024 logD1 | 2436.15 | dial | gpu-frontier | 0.88 | 0.95 | never | 26.67 |
| C | grid 1024x1024 logD2 | 2218.73 | dial | gpu-nearfar | 0.52 | 0.55 | never | 1.86 |
| C | grid 1024x1024 logD4 | 1885.75 | dial | gpu-nearfar | 0.46 | 0.48 | never | 45.22 |
| C | grid 1024x1024 logD6 | 1657.51 | dial | gpu-frontier | 0.48 | 0.49 | never | 244.27 |
| C | grid 1024x1024 u[1,100000] | 2382.27 | delta | gpu-frontier | 1.10 | 1.16 | 1.00 | 45.97 |
| C | grid 1024x1024 u[1,100] | 2391.44 | dial | gpu-frontier | 0.68 | 0.71 | never | 44.33 |
| C | grid 1024x1024 unit | 2472.96 | dial | gpu-frontier | 1.48 | 1.73 | 1.00 | 1.00 |
| C | uniform n=1048576 logD1 | 493445.41 | delta-omp | gpu-nearfar | 1.76 | 5.80 | 1.00 | 1.00 |
| C | uniform n=1048576 logD2 | 349523.83 | delta-omp | gpu-nearfar | 2.17 | 7.21 | 1.00 | 1.42 |
| C | uniform n=1048576 logD4 | 226718.16 | delta-omp | gpu-frontier | 7.11 | 13.66 | 1.00 | 5.84 |
| C | uniform n=1048576 logD6 | 172978.63 | delta-omp | gpu-frontier | 8.43 | 14.29 | 1.00 | 7.56 |
| C | uniform n=1048576 u[1,100000] | 270599.10 | delta-omp | gpu-nearfar | 1.77 | 4.86 | 1.00 | 1.06 |
| C | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.78 | 4.85 | 1.00 | 1.04 |
| C | uniform n=1048576 unit | 838857.20 | delta-omp | gpu-frontier | 1.63 | 8.22 | 1.00 | 1.00 |
| E | grid 1024x1024 u[1,100] | 2554.82 | dial | gpu-frontier | 0.70 | 0.74 | never | 42.45 |
| E | grid 1024x512 u[1,100] | 1656.56 | dial | gpu-frontier | 0.44 | 0.47 | never | 36.39 |
| E | grid 2048x1024 u[1,100] | 3139.29 | dial | gpu-frontier | 0.66 | 0.68 | never | 57.28 |
| E | grid 2048x2048 u[1,100] | 4821.38 | dial | gpu-nearfar | 1.10 | 1.16 | 1.00 | 1.03 |
| E | grid 256x256 u[1,100] | 624.75 | dial | gpu-frontier | 0.26 | 0.28 | never | 11.29 |
| E | grid 4096x2048 u[1,100] | 6417.19 | dial | gpu-nearfar | 1.48 | 1.58 | 1.00 | 1.03 |
| E | grid 4096x4096 u[1,100] | 10886.52 | dial | gpu-nearfar | 2.10 | 2.32 | 1.00 | 1.03 |
| E | grid 512x256 u[1,100] | 739.64 | dial | gpu-frontier | 0.30 | 0.31 | never | 20.56 |
| E | grid 512x512 u[1,100] | 1362.45 | dial | gpu-frontier | 0.54 | 0.59 | never | 17.53 |
| E | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.73 | 4.82 | 1.00 | 1.04 |
| E | uniform n=131072 u[1,100] | 41135.11 | delta-omp | gpu-frontier | 1.50 | 3.46 | 1.00 | 3.32 |
| E | uniform n=16777216 u[1,100] | 3781529.87 | delta-omp | gpu-nearfar | 2.82 | 5.31 | 1.00 | 1.04 |
| E | uniform n=2097152 u[1,100] | 532743.18 | delta-omp | gpu-nearfar | 2.13 | 7.32 | 1.00 | 1.04 |
| E | uniform n=262144 u[1,100] | 82271.16 | delta-omp | gpu-frontier | 1.45 | 3.76 | 1.00 | 3.30 |
| E | uniform n=4194304 u[1,100] | 1016799.39 | delta-omp | gpu-nearfar | 2.55 | 7.84 | 1.00 | 1.04 |
| E | uniform n=524288 u[1,100] | 147212.96 | delta-omp | gpu-frontier | 1.43 | 3.48 | 1.00 | 3.57 |
| E | uniform n=65536 u[1,100] | 21843.92 | delta-omp | gpu-frontier | 1.82 | 3.82 | 1.00 | 3.26 |
| E | uniform n=8388608 u[1,100] | 1917394.57 | delta-omp | gpu-nearfar | 2.61 | 6.15 | 1.00 | 1.04 |
| HO | geometric n=1048576 u[1,100] | 5585.35 | dial | gpu-nearfar | 0.70 | 0.74 | never | 1.05 |
| HO | grid 1024x1024 natural u[1,100] | 2550.48 | dial | gpu-frontier | 0.34 | 0.36 | never | 44.89 |
| HO | grid 1024x1024 u[1,100] | 2391.44 | dial | gpu-frontier | 0.68 | 0.72 | never | 44.33 |
| HO | rmat n=1048576 u[1,100] | 681132.67 | delta-omp | gpu-frontier | 1.32 | 3.50 | 1.00 | 3.29 |
| HO | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.75 | 4.84 | 1.00 | 1.04 |
