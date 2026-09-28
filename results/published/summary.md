# Summary

13th Gen Intel(R) Core(TM) i9-13900KF, NVIDIA GeForce RTX 4090, driver 575.57.08. 27246 timed runs over 62 graphs, plus 803 profiling and placement runs.

## Calibration constants

| constant | value |
|---|---|
| alpha_frontier_us | 15.352 |
| alpha_nearfar_us_per_sync | 9.216 |
| alpha_sync_us | 3.864 |
| theta_uniform_deg8_edges_per_ns | 4.484 |
| cost_per_vertex_uniform_deg8_ns | 1.784 |
| theta_grid_deg4_edges_per_ns | 1.999 |
| cost_per_vertex_grid_deg4_ns | 2.000 |
| lambda_gpu-topo_ns | 132.931 |
| lambda_gpu-frontier_ns | 17.947 |
| d2h_pageable_1MB_GBps | 8.603 |
| d2h_pageable_16MB_GBps | 14.668 |
| d2h_pageable_1024MB_GBps | 15.365 |
| d2h_pinned_1MB_GBps | 16.442 |
| d2h_pinned_16MB_GBps | 19.044 |
| d2h_pinned_1024MB_GBps | 19.368 |
| h2d_pageable_1MB_GBps | 12.209 |
| h2d_pageable_16MB_GBps | 16.178 |
| h2d_pageable_1024MB_GBps | 16.729 |
| h2d_pinned_1MB_GBps | 17.430 |
| h2d_pinned_16MB_GBps | 20.043 |
| h2d_pinned_1024MB_GBps | 20.366 |
| omp_region_24_threads_us | 2.099 |

## Model (fitted on A, median error elsewhere)

| solver | model | alpha_us | ns_per_edge | ns_per_vertex | error_A | error_B | error_C | error_E | error_HO |
|---|---|---|---|---|---|---|---|---|---|
| gpu-frontier | syncs + edges | 15.987 | 0.064 |  | 0.258 | 0.452 | 0.432 | 0.414 | 0.415 |
| gpu-frontier | syncs + vertices + edges | 13.410 | 0.015 | 0.845 | 0.030 | 0.374 | 0.048 | 0.132 | 0.061 |
| gpu-nearfar | syncs + edges | 5.634 | 0.121 |  | 0.071 |  | 0.080 | 0.081 | 0.078 |
| gpu-nearfar | syncs + vertices + edges | 5.668 | 0.026 | 0.786 | 0.057 |  | 0.068 | 0.072 | 0.070 |
| gpu-frontier | calibration | 15.352 | 0.223 | 1.784 |  |  |  |  |  |

## Picking the faster device

| rule | accuracy | regret | cases |
|---|---|---|---|
| always CPU | 0.365 | 0.960 | 260 |
| always GPU | 0.635 | 0.240 | 260 |
| BFS frontier width, fitted | 0.862 | 0.076 | 260 |
| graph properties and calibrated constants | 0.888 | 0.048 | 260 |
| model on the run's own counters, true CPU time | 0.992 | 0.000 | 260 |

## Break even query counts

| label | gpu_solver | best_cpu | q_cpu_ms | q_gpu_ms | U_ms | Kstar_measured | Kstar_predicted | q_batch_ms | Kstar_vs_batch |
|---|---|---|---|---|---|---|---|---|---|
| layered W=256 u[1,100] | gpu-frontier | dial | 98.31 | 166.03 | 14.07 | never | never | 8.48 | never |
| layered W=256 u[1,100] | gpu-nearfar | dial | 98.31 | 279.92 | 4.69 | never | never | 8.48 | never |
| layered W=4096 u[1,100] | gpu-frontier | delta-omp | 36.77 | 12.44 | 5.43 | 1.00 | 1.00 | 11.74 | never |
| layered W=4096 u[1,100] | gpu-nearfar | delta-omp | 36.77 | 24.27 | 4.51 | 1.00 | 1.00 | 11.74 | never |
| layered W=65536 u[1,100] | gpu-frontier | delta-omp | 21.49 | 4.15 | 5.50 | 1.00 | 1.00 | 16.78 | 1.00 |
| layered W=65536 u[1,100] | gpu-nearfar | delta-omp | 21.49 | 5.46 | 4.40 | 1.00 | 1.00 | 16.78 | 1.00 |
| uniform n=1048576 u[1,100] | gpu-frontier | delta-omp | 21.76 | 4.27 | 5.06 | 1.00 | 1.00 | 17.60 | 1.00 |
| uniform n=1048576 u[1,100] | gpu-nearfar | delta-omp | 21.76 | 5.05 | 4.99 | 1.00 | 1.00 | 17.60 | 1.00 |
| grid 1024x1024 u[1,100] | gpu-frontier | dial | 47.23 | 61.90 | 1.93 | never | never | 5.09 | never |
| grid 1024x1024 u[1,100] | gpu-nearfar | dial | 47.23 | 130.76 | -4.45 | never | never | 5.09 | never |

## Hardware counters

| kernel | launches | kernel_us | sm_active_pct | occupancy_pct | sm_tail | dram_pct | l2_pct | issue_active_pct | lanes_active | sectors_per_request | bottleneck |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid_frontier | 40 | 3.36 | 64.69 | 15.80 | 1.16 | 3.41 | 6.57 | 6.18 | 24.95 | 1.12 | too little parallelism |
| hub_frontier | 22 | 5.57 | 58.30 | 46.55 | 50.20 | 6.88 | 12.25 | 6.99 | 26.13 | 1.29 | load imbalance |
| hub_topo | 16 | 10472.58 | 2.81 | 60.96 | 38.44 | 0.63 | 1.23 | 4.22 | 18.04 | 12.02 | load imbalance |
| layered_frontier | 40 | 15.31 | 89.67 | 72.92 | 1.06 | 10.77 | 17.91 | 9.27 | 25.61 | 1.29 | memory latency |
| uniform_edge | 19 | 637.50 | 100.53 | 87.74 | 1.00 | 58.23 | 67.01 | 5.17 | 31.86 | 3.32 | bandwidth |
| uniform_frontier | 31 | 42.40 | 100.46 | 83.33 | 1.04 | 9.97 | 9.40 | 3.64 | 25.00 | 1.29 | memory latency |
| uniform_nearfar | 60 | 3.65 | 101.11 | 77.95 | 1.92 | 11.43 | 10.31 | 4.13 | 24.66 | 1.32 | memory latency |
| uniform_topo | 21 | 802.34 | 100.07 | 86.95 | 1.00 | 30.53 | 58.15 | 4.06 | 18.88 | 13.66 | memory latency |

Nsight Systems, grid, gpu-frontier, 4 solves: 35.3 ms in kernels, 97.9 ms in 6009 blocking copies.

| solver | graph | ms_per_solve | ipc_p_core | ipc_e_core | e_core_share | llc_misses_per_1k_instructions | thread_utilization |
|---|---|---|---|---|---|---|---|
| delta-omp | grid_unit | 50.89 | 0.78 | 0.49 | 0.32 | 2.14 | 0.04 |
| delta-omp | uniform_2e22 | 115.87 | 0.25 | 0.16 | 0.34 | 9.08 | 0.90 |
| dial | grid_unit | 39.74 | 0.59 | 0.56 | 0.42 | 2.93 | 0.98 |
| dial | uniform_2e22 | 658.69 | 0.24 | 0.60 | 0.62 | 8.30 | 1.00 |
| dijkstra | grid_unit | 130.06 | 0.72 | 1.05 | 0.59 | 0.91 | 1.00 |
| dijkstra | uniform_2e22 | 1745.55 | 0.49 | 0.56 | 0.44 | 4.01 | 1.00 |

## Serial CPU solvers pinned to one P core

| graph | solver | pcore | unpinned | unpinned_over_pinned |
|---|---|---|---|---|
| grid | delta | 78.94 | 77.43 | 0.98 |
| grid | dial | 47.80 | 46.77 | 0.98 |
| grid | dijkstra | 154.47 | 151.28 | 0.98 |
| layered | delta | 152.27 | 151.10 | 0.99 |
| layered | dial | 98.14 | 97.39 | 0.99 |
| layered | dijkstra | 283.80 | 282.48 | 1.00 |
| uniform | delta | 175.17 | 173.98 | 0.99 |
| uniform | dial | 101.44 | 100.75 | 0.99 |
| uniform | dijkstra | 313.35 | 313.54 | 1.00 |

## Per graph

| experiment | label | Pi | best_cpu | best_gpu | S1 | Sinf | Kstar | iota | delta |
|---|---|---|---|---|---|---|---|---|---|
| A | layered W=1024 k=16 u[1,100] | 29638.96 | dial | gpu-frontier | 3.09 | 4.43 | 1.00 | 25.44 | 1.00 |
| A | layered W=1024 natural u[1,100] | 7187.33 | dial | gpu-frontier | 0.64 | 0.73 | never | 31.79 | 1.00 |
| A | layered W=1024 u[1,100] | 7187.33 | dial | gpu-frontier | 1.77 | 2.00 | 1.00 | 31.79 | 1.00 |
| A | layered W=1024 unit | 8176.02 | dial | gpu-frontier | 3.85 | 5.45 | 1.00 | 1.00 | 1.00 |
| A | layered W=16 u[1,100] | 117.76 | dial | gpu-frontier | 0.05 | 0.05 | never | 1386.39 | 1.00 |
| A | layered W=16 unit | 128.00 | dial | gpu-frontier | 0.09 | 0.09 | never | 1.00 | 1.00 |
| A | layered W=16384 k=16 u[1,100] | 377499.69 | delta-omp | gpu-frontier | 2.26 | 9.10 | 1.00 | 3.66 | 1.01 |
| A | layered W=16384 natural u[1,100] | 92781.30 | delta-omp | gpu-frontier | 1.83 | 3.86 | 1.00 | 3.71 | 1.01 |
| A | layered W=16384 u[1,100] | 92781.30 | delta-omp | gpu-frontier | 2.55 | 5.41 | 1.00 | 3.71 | 1.01 |
| A | layered W=16384 unit | 127039.02 | delta-omp | gpu-frontier | 1.40 | 4.98 | 1.00 | 1.00 | 1.02 |
| A | layered W=256 u[1,100] | 1827.93 | dial | gpu-frontier | 0.54 | 0.56 | never | 112.55 | 1.00 |
| A | layered W=256 unit | 2047.50 | dial | gpu-frontier | 1.24 | 1.37 | 1.00 | 1.00 | 1.00 |
| A | layered W=262144 u[1,100] | 202950.19 | delta-omp | gpu-nearfar | 1.72 | 4.02 | 1.00 | 1.04 | 11.15 |
| A | layered W=262144 unit | 571950.55 | delta-omp | gpu-frontier | 1.54 | 6.49 | 1.00 | 1.00 | 1.09 |
| A | layered W=4096 u[1,100] | 27085.45 | delta-omp | gpu-frontier | 1.81 | 2.66 | 1.00 | 8.57 | 1.00 |
| A | layered W=4096 unit | 32513.00 | delta-omp | gpu-frontier | 6.30 | 15.73 | 1.00 | 1.00 | 1.00 |
| A | layered W=64 k=16 u[1,100] | 1889.74 | dial | gpu-frontier | 0.32 | 0.33 | never | 322.30 | 1.00 |
| A | layered W=64 natural u[1,100] | 459.50 | dial | gpu-frontier | 0.05 | 0.05 | never | 459.69 | 1.00 |
| A | layered W=64 u[1,100] | 459.50 | dial | gpu-frontier | 0.15 | 0.15 | never | 459.69 | 1.00 |
| A | layered W=64 unit | 512.00 | dial | gpu-frontier | 0.34 | 0.35 | never | 1.00 | 1.00 |
| A | layered W=65536 u[1,100] | 224694.86 | delta-omp | gpu-frontier | 1.83 | 4.44 | 1.00 | 3.13 | 1.03 |
| A | layered W=65536 unit | 462607.06 | delta-omp | gpu-frontier | 1.52 | 6.90 | 1.00 | 1.00 | 1.06 |
| C | grid 1024x1024 logD1 | 2436.15 | dial | gpu-frontier | 0.89 | 0.96 | never | 26.67 | 1.00 |
| C | grid 1024x1024 logD2 | 2218.73 | dial | gpu-nearfar | 0.52 | 0.55 | never | 1.86 | 7.14 |
| C | grid 1024x1024 logD4 | 1885.75 | dial | gpu-nearfar | 0.46 | 0.48 | never | 45.10 | 5.18 |
| C | grid 1024x1024 logD6 | 1657.51 | dial | gpu-frontier | 0.49 | 0.49 | never | 244.27 | 1.00 |
| C | grid 1024x1024 u[1,100000] | 2382.27 | delta | gpu-frontier | 1.11 | 1.17 | 1.00 | 45.97 | 1.00 |
| C | grid 1024x1024 u[1,100] | 2391.44 | dial | gpu-frontier | 0.68 | 0.72 | never | 44.33 | 1.00 |
| C | grid 1024x1024 unit | 2472.96 | dial | gpu-frontier | 1.47 | 1.72 | 1.00 | 1.00 | 1.00 |
| C | uniform n=1048576 logD1 | 493445.41 | delta-omp | gpu-nearfar | 1.76 | 5.88 | 1.00 | 1.00 | 14.71 |
| C | uniform n=1048576 logD2 | 349523.83 | delta-omp | gpu-nearfar | 2.22 | 7.32 | 1.00 | 1.42 | 8.58 |
| C | uniform n=1048576 logD4 | 226718.16 | delta-omp | gpu-frontier | 7.13 | 13.56 | 1.00 | 5.84 | 0.97 |
| C | uniform n=1048576 logD6 | 172978.63 | delta-omp | gpu-frontier | 8.37 | 14.26 | 1.00 | 7.57 | 0.98 |
| C | uniform n=1048576 u[1,100000] | 270599.10 | delta-omp | gpu-nearfar | 1.80 | 4.95 | 1.00 | 1.06 | 12.39 |
| C | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.79 | 4.92 | 1.00 | 1.04 | 13.00 |
| C | uniform n=1048576 unit | 838857.20 | delta-omp | gpu-frontier | 1.63 | 8.28 | 1.00 | 1.00 | 1.10 |
| E | grid 1024x1024 u[1,100] | 2554.82 | dial | gpu-frontier | 0.70 | 0.74 | never | 42.45 | 1.00 |
| E | grid 1024x512 u[1,100] | 1656.56 | dial | gpu-frontier | 0.45 | 0.47 | never | 36.39 | 1.00 |
| E | grid 2048x1024 u[1,100] | 3139.29 | dial | gpu-frontier | 0.68 | 0.70 | never | 57.28 | 1.00 |
| E | grid 2048x2048 u[1,100] | 4821.38 | dial | gpu-nearfar | 1.11 | 1.17 | 1.00 | 1.03 | 13.47 |
| E | grid 256x256 u[1,100] | 624.75 | dial | gpu-frontier | 0.26 | 0.27 | never | 11.29 | 1.00 |
| E | grid 4096x2048 u[1,100] | 6417.19 | dial | gpu-nearfar | 1.50 | 1.61 | 1.00 | 1.03 | 13.70 |
| E | grid 4096x4096 u[1,100] | 10886.52 | dial | gpu-nearfar | 2.11 | 2.34 | 1.00 | 1.03 | 14.17 |
| E | grid 512x256 u[1,100] | 739.64 | dial | gpu-frontier | 0.29 | 0.31 | never | 20.56 | 1.00 |
| E | grid 512x512 u[1,100] | 1362.45 | dial | gpu-frontier | 0.53 | 0.58 | never | 17.53 | 1.00 |
| E | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.78 | 4.91 | 1.00 | 1.04 | 12.98 |
| E | uniform n=131072 u[1,100] | 41135.11 | delta-omp | gpu-frontier | 1.50 | 3.46 | 1.00 | 3.33 | 1.02 |
| E | uniform n=16777216 u[1,100] | 3781529.87 | delta-omp | gpu-nearfar | 2.82 | 5.38 | 1.00 | 1.04 | 13.35 |
| E | uniform n=2097152 u[1,100] | 532743.18 | delta-omp | gpu-nearfar | 2.16 | 7.48 | 1.00 | 1.04 | 13.16 |
| E | uniform n=262144 u[1,100] | 82271.16 | delta-omp | gpu-frontier | 1.42 | 3.69 | 1.00 | 3.30 | 1.00 |
| E | uniform n=4194304 u[1,100] | 1016799.39 | delta-omp | gpu-nearfar | 2.57 | 7.97 | 1.00 | 1.04 | 13.45 |
| E | uniform n=524288 u[1,100] | 147212.96 | delta-omp | gpu-frontier | 1.41 | 3.41 | 1.00 | 3.56 | 1.00 |
| E | uniform n=65536 u[1,100] | 21843.92 | delta-omp | gpu-frontier | 1.82 | 3.80 | 1.00 | 3.26 | 1.02 |
| E | uniform n=8388608 u[1,100] | 1917394.57 | delta-omp | gpu-nearfar | 2.62 | 5.99 | 1.00 | 1.04 | 12.77 |
| HO | geometric n=1048576 u[1,100] | 5585.35 | dial | gpu-nearfar | 0.70 | 0.74 | never | 1.05 | 15.50 |
| HO | grid 1024x1024 natural u[1,100] | 2550.48 | dial | gpu-frontier | 0.34 | 0.35 | never | 44.89 | 1.00 |
| HO | grid 1024x1024 u[1,100] | 2391.44 | dial | gpu-frontier | 0.69 | 0.73 | never | 44.33 | 1.00 |
| HO | rmat n=1048576 u[1,100] | 681132.67 | delta-omp | gpu-frontier | 1.33 | 3.53 | 1.00 | 3.28 | 1.08 |
| HO | uniform n=1048576 u[1,100] | 284440.09 | delta-omp | gpu-nearfar | 1.82 | 4.99 | 1.00 | 1.04 | 12.97 |
