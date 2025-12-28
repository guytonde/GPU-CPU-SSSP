#!/usr/bin/env bash
set -euo pipefail

# Benchmark script: generates graphs of varying sizes and runs CPU/GPU algorithms multiple times.
# Outputs a CSV at tools/benchmark_results.csv with columns:
# V, E, algorithm, cpu_time_us, gpu_time_us, diffs

BIN_CPU=bin/sssp_cpu
BIN_GPU=bin/sssp_gpu
GEN=bin/generate_graph

# sizes (space-separated), can override via SIZES env var
: ${SIZES:="100 500 1000 5000 10000 50000 100000 200000"}
: ${RUNS:=3}
: ${EDGES_FACTOR:=5} # edges = V * EDGES_FACTOR

OUT_CSV=tools/benchmark_results.csv
mkdir -p tools

echo "V,E,algorithm,cpu_time_us,gpu_time_us,diffs" > $OUT_CSV

echo "Building binaries..."
make cpu gpu >/dev/null

for V in $SIZES; do
  # adapt edges factor for large graphs to avoid generator limits
  local_factor=$EDGES_FACTOR
  if [ "$V" -ge 200000 ]; then
    local_factor=0  # very sparse: just E = V
  elif [ "$V" -ge 50000 ]; then
    local_factor=1
  elif [ "$V" -ge 10000 ]; then
    local_factor=2
  fi
  E=$((V * local_factor))
  echo "Generating graph V=$V E=$E..."
  ./${GEN} ${V} ${E} 1 100 graphs/generated/bench_graph_${V}.txt

  for alg in dijkstra bellman-ford bfs; do
    echo "Benchmarking $alg on V=$V..."
    cpu_times=()
    gpu_times=()
    diffs=0

    for run in $(seq 1 $RUNS); do
      cpu_out=$(mktemp)
      gpu_out=$(mktemp)

      ./${BIN_CPU} $alg graphs/generated/bench_graph_${V}.txt $V 0 > $cpu_out
      if [ "$alg" = "dijkstra" ]; then galg="gpu-dijkstra"; elif [ "$alg" = "bellman-ford" ]; then galg="gpu-bellman-ford"; else galg="gpu-bfs"; fi
      ./${BIN_GPU} $galg graphs/generated/bench_graph_${V}.txt $V 0 > $gpu_out

      ctime=$(grep -E "Time: " $cpu_out | tail -n1 | awk '{print $2+0}')
      gtime=$(grep -E "Time: " $gpu_out | tail -n1 | awk '{print $2+0}')
      cpu_times+=("$ctime")
      gpu_times+=("$gtime")

      # compare distances
      grep '^Vertex ' $cpu_out | awk -F": " '{print $2}' > $cpu_out.dist
      grep '^Vertex ' $gpu_out | awk -F": " '{print $2}' > $gpu_out.dist
      n_cpu=$(wc -l < $cpu_out.dist)
      n_gpu=$(wc -l < $gpu_out.dist)
      if [ $n_cpu -gt $n_gpu ]; then seq $n_cpu | xargs -I{} echo INF >> $gpu_out.dist; fi
      if [ $n_gpu -gt $n_cpu ]; then seq $n_gpu | xargs -I{} echo INF >> $cpu_out.dist; fi
      run_diffs=$(paste $cpu_out.dist $gpu_out.dist | awk '{if($1!=$2) c++} END{print c+0}')
      diffs=$((diffs + run_diffs))

      rm -f $cpu_out $gpu_out $cpu_out.dist $gpu_out.dist
    done

    # compute median times
    median() {
      arr=("$@")
      IFS=$'\n' sorted=($(sort -n <<<"${arr[*]}"))
      unset IFS
      len=${#sorted[@]}
      if [ $((len % 2)) -eq 1 ]; then
        echo ${sorted[$((len/2))]}
      else
        a=${sorted[$((len/2 - 1))]}
        b=${sorted[$((len/2))]}
        echo $(((a + b) / 2))
      fi
    }

    cpu_med=$(median "${cpu_times[@]}")
    gpu_med=$(median "${gpu_times[@]}")

    echo "$V,$E,$alg,$cpu_med,$gpu_med,$diffs" >> $OUT_CSV
    echo "  $alg: cpu=$cpu_med us, gpu=$gpu_med us, total_diffs=$diffs"
  done
done

echo "Benchmark complete. Results: $OUT_CSV"

echo
echo "Summary table (also saved to $OUT_CSV):"
printf "%6s %8s %18s %12s %12s %6s\n" "V" "E" "Algorithm" "CPU(us)" "GPU(us)" "Diffs"
printf "%6s %8s %18s %12s %12s %6s\n" "------" "--------" "------------------" "----------" "----------" "------"
tail -n +2 $OUT_CSV | while IFS=, read -r v e alg cpu gpu dif; do
  printf "%6s %8s %18s %12s %12s %6s\n" "$v" "$e" "$alg" "$cpu" "$gpu" "$dif"
done

