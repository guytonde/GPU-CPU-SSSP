#!/usr/bin/env bash
# Experiment F: hardware counters for a few kernels and CPU solvers, and a CPU
# placement control. ncu and perf need sudo on this machine; nothing else runs
# as root. Run it as your own user, or with SUDO_PWFILE pointing at a file only
# you can read. Writes results/raw/F.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/raw/F
G=graphs/profile
NCU=${NCU:-$(command -v ncu || echo /usr/local/cuda/bin/ncu)}
NSYS=${NSYS:-$(command -v nsys || echo /usr/local/cuda/bin/nsys)}
PERF=${PERF:-$(command -v perf)}
mkdir -p "$OUT" "$G"

as_root() {
    if [ -n "${SUDO_PWFILE:-}" ]; then
        sudo -S -p '' "$@" < "$SUDO_PWFILE"
    else
        sudo "$@"
    fi
}

if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
    echo "another process is on the GPU; not profiling" >&2
    exit 1
fi

bin/gen_graph --topo grid --rows 1024 --cols 1024 --wdist unit --out $G/grid_unit.bin
bin/gen_graph --topo layered --n 1048576 --layer-width 256 --out $G/layered_W256.bin
bin/gen_graph --topo uniform --n 4194304 --m 16777216 --out $G/uniform_2e22.bin
bin/gen_graph --topo hub --n 1048576 --m 4194304 --hub-fraction 0.015625 --out $G/hub.bin

# Caches and clocks are left as they are in the experiments.
METRICS=gpu__time_duration.sum,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__cycles_active.max,sm__cycles_active.avg,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__thread_inst_executed_per_inst_executed.ratio,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum

# name graph solver kernel launches-to-skip launches-to-capture
profile() {
    if [ -s "$OUT/ncu_$1.csv" ]; then echo "== ncu $1 (done)"; return; fi
    echo "== ncu $1"
    as_root "$NCU" --cache-control none --clock-control none --metrics "$METRICS" \
        --kernel-name "$4" --launch-skip "$5" --launch-count "$6" --csv \
        --log-file "$OUT/ncu_$1.csv" \
        bin/bench "$2" --only "$3" --sources 1 --reps 1 --warmup 0 --no-preheat --quiet \
        --experiment F --out "$OUT/bench" > /dev/null
}
profile grid_frontier     $G/grid_unit.bin      gpu-frontier expand          400 40
profile layered_frontier  $G/layered_W256.bin   gpu-frontier expand          1000 40
profile uniform_frontier  $G/uniform_2e22.bin   gpu-frontier expand          0 60
profile uniform_nearfar   $G/uniform_2e22.bin   gpu-nearfar  expand_split    20 60
profile uniform_edge      $G/uniform_2e22.bin   gpu-edge     relax_by_edge   0 60
profile uniform_topo      $G/uniform_2e22.bin   gpu-topo     relax_by_vertex 0 60
profile hub_topo          $G/hub.bin            gpu-topo     relax_by_vertex 0 60
profile hub_frontier      $G/hub.bin            gpu-frontier expand          0 60

# Give the files ncu wrote as root back to the user.
as_root chown -R "$(id -u):$(id -g)" "$OUT"

# Timeline of the grid, where synchronization dominates.
echo "== nsys grid_frontier"
"$NSYS" profile --trace=cuda --force-overwrite true -o "$OUT/nsys_grid_frontier" \
    bin/bench $G/grid_unit.bin --only gpu-frontier --sources 1 --reps 3 --warmup 1 --quiet \
    --experiment F --out "$OUT/bench" > /dev/null
"$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
    --output "$OUT/nsys_grid_frontier" "$OUT/nsys_grid_frontier.nsys-rep" > /dev/null
rm -f "$OUT"/nsys_grid_frontier.nsys-rep "$OUT"/nsys_grid_frontier.sqlite

# 11 repetitions minus 1 leaves 10 solves without loading and checking.
EVENTS=duration_time,task-clock,cycles,instructions,LLC-load-misses,branch-misses
for graph in uniform_2e22 grid_unit; do
    for s in dijkstra dial delta-omp; do
        for reps in 1 11; do
            if [ -s "$OUT/perf_${s}_${graph}_r${reps}.csv" ]; then continue; fi
            echo "== perf $s $graph reps=$reps"
            as_root "$PERF" stat -x, -e "$EVENTS" -o "$OUT/perf_${s}_${graph}_r${reps}.csv" \
                bin/bench $G/$graph.bin --only $s --sources 1 --reps $reps --warmup 0 --quiet \
                --no-oracle --experiment F --out "$OUT/bench" > /dev/null
        done
    done
done

as_root chown -R "$(id -u):$(id -g)" "$OUT"

# Serial CPU solvers pinned to one P core (CPU 2 on the i9 13900KF) and unpinned.
PCORE=${PCORE:-2}
if [ ! -d "$OUT/placement" ]; then
    mkdir -p $G/placement
    bin/gen_graph --topo grid --rows 1024 --cols 1024 --out $G/placement/grid.bin
    bin/gen_graph --topo layered --n 1048576 --layer-width 1024 --out $G/placement/layered.bin
    bin/gen_graph --topo uniform --n 1048576 --m 4194304 --out $G/placement/uniform.bin
    for graph in grid layered uniform; do
        echo "== placement $graph"
        args=($G/placement/$graph.bin --only dijkstra,dial,delta --sources 8 --reps 5
              --experiment F --out "$OUT/placement" --quiet)
        bin/bench "${args[@]}" --affinity-label unpinned
        taskset -c "$PCORE" bin/bench "${args[@]}" --affinity-label pcore
    done
fi
rm -rf "$G"
echo "wrote $OUT"
