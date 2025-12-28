#!/usr/bin/env bash
set -euo pipefail

# 3D Benchmark: varies vertices (V), edges (E), and weight ranges independently.
# Tests combinations of: V x E_factor x weight_range
# Outputs: tools/benchmark_3d_results.csv

BIN_CPU=bin/sssp_cpu
BIN_GPU=bin/sssp_gpu
GEN=bin/generate_graph

# Parameters: each can be overridden via env
: ${VERTICES:="100 500 1000 2000 5000"}
: ${EDGE_FACTORS:="2 5 10 15 20"}  # E = V * factor
: ${WEIGHT_RANGES:="1_10 1_100 1_1000 1_10000"}  # min_max
: ${RUNS:=1}

OUT_CSV=tools/benchmark_3d_results.csv
mkdir -p tools graphs/generated

echo "V,E,min_weight,max_weight,algorithm,cpu_time_us,gpu_time_us,diffs" > $OUT_CSV

echo "Building binaries..."
make cpu gpu >/dev/null

total_tests=0
completed=0

for V in $VERTICES; do
  for EF in $EDGE_FACTORS; do
    for WR in $WEIGHT_RANGES; do
      # parse weight range ("1_100" -> min=1, max=100)
      IFS='_' read -r MINW MAXW <<< "$WR"
      E=$((V * EF))
      
      # clamp edges to max possible for this V
      MAX_EDGES=$((V * (V - 1) / 2))
      if [ $E -gt $MAX_EDGES ]; then
        continue  # skip infeasible combinations
      fi

      total_tests=$((total_tests + 1))
    done
  done
done

for V in $VERTICES; do
  for EF in $EDGE_FACTORS; do
    for WR in $WEIGHT_RANGES; do
      IFS='_' read -r MINW MAXW <<< "$WR"
      E=$((V * EF))
      MAX_EDGES=$((V * (V - 1) / 2))
      
      if [ $E -gt $MAX_EDGES ]; then
        continue
      fi

      completed=$((completed + 1))
      echo "[$completed/$total_tests] Generating V=$V E=$E weights=[$MINW..$MAXW]..."
      GRAPH_FILE="graphs/generated/3d_V${V}_E${E}_W${MINW}_${MAXW}.txt"
      ./${GEN} ${V} ${E} ${MINW} ${MAXW} "$GRAPH_FILE"

      for alg in dijkstra bellman-ford bfs; do
        echo "  Benchmarking $alg..."
        cpu_times=()
        gpu_times=()
        diffs=0

        for run in $(seq 1 $RUNS); do
          cpu_out=$(mktemp)
          gpu_out=$(mktemp)

          ./${BIN_CPU} $alg "$GRAPH_FILE" $V 0 > $cpu_out 2>&1 || true
          if [ "$alg" = "dijkstra" ]; then galg="gpu-dijkstra"; elif [ "$alg" = "bellman-ford" ]; then galg="gpu-bellman-ford"; else galg="gpu-bfs"; fi
          ./${BIN_GPU} $galg "$GRAPH_FILE" $V 0 > $gpu_out 2>&1 || true

          ctime=$(grep -E "Time: " $cpu_out | tail -n1 | awk '{print $2+0}' || echo 0)
          gtime=$(grep -E "Time: " $gpu_out | tail -n1 | awk '{print $2+0}' || echo 0)
          cpu_times+=("$ctime")
          gpu_times+=("$gtime")

          # compare distances
          grep '^Vertex ' $cpu_out | awk -F": " '{print $2}' > $cpu_out.dist 2>/dev/null || echo ""
          grep '^Vertex ' $gpu_out | awk -F": " '{print $2}' > $gpu_out.dist 2>/dev/null || echo ""
          n_cpu=$(wc -l < $cpu_out.dist 2>/dev/null || echo 0)
          n_gpu=$(wc -l < $gpu_out.dist 2>/dev/null || echo 0)
          
          if [ $n_cpu -gt 0 ] && [ $n_gpu -gt 0 ]; then
            if [ $n_cpu -gt $n_gpu ]; then seq $n_cpu | xargs -I{} echo INF >> $gpu_out.dist; fi
            if [ $n_gpu -gt $n_cpu ]; then seq $n_gpu | xargs -I{} echo INF >> $cpu_out.dist; fi
            run_diffs=$(paste $cpu_out.dist $gpu_out.dist 2>/dev/null | awk '{if($1!=$2) c++} END{print c+0}' || echo 0)
            diffs=$((diffs + run_diffs))
          fi

          rm -f $cpu_out $gpu_out $cpu_out.dist $gpu_out.dist
        done

        # compute medians
        median() {
          arr=("$@")
          IFS=$'\n' sorted=($(sort -n <<<"${arr[*]}"))
          unset IFS
          len=${#sorted[@]}
          if [ $len -eq 0 ]; then echo 0; return; fi
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

        echo "$V,$E,$MINW,$MAXW,$alg,$cpu_med,$gpu_med,$diffs" >> $OUT_CSV
      done
    done
  done
done

echo "Benchmark complete. Results: $OUT_CSV"
echo ""
echo "Summary (sorted by V, E, weight range):"
printf "%6s %6s %8s %8s %18s %12s %12s %6s\n" "V" "E" "MinW" "MaxW" "Algorithm" "CPU(us)" "GPU(us)" "Diffs"
printf "%6s %6s %8s %8s %18s %12s %12s %6s\n" "------" "------" "--------" "--------" "------------------" "----------" "----------" "------"
tail -n +2 $OUT_CSV | sort -t, -k1,1n -k2,2n -k3,3n | while IFS=, read -r v e minw maxw alg cpu gpu dif; do
  printf "%6s %6s %8s %8s %18s %12s %12s %6s\n" "$v" "$e" "$minw" "$maxw" "$alg" "$cpu" "$gpu" "$dif"
done
