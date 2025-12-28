#!/bin/sh
set -e
ALG_LIST="dijkstra bellman-ford bfs"
printf "%-20s | %-12s | %-12s | %-8s\n" "Algorithm" "CPU Time(us)" "GPU Time(us)" "Diffs"
printf "%-20s-+-%-12s-+-%-12s-+-%-8s\n" "--------------------" "------------" "------------" "--------"
for alg in $ALG_LIST; do
  cpu_out=$(mktemp); gpu_out=$(mktemp)
  ./bin/sssp_cpu $alg graphs/generated/test_graph.txt 100 0 > $cpu_out
  if [ "$alg" = "dijkstra" ]; then galg="gpu-dijkstra"; elif [ "$alg" = "bellman-ford" ]; then galg="gpu-bellman-ford"; else galg="gpu-bfs"; fi
  ./bin/sssp_gpu $galg graphs/generated/test_graph.txt 100 0 > $gpu_out
  cpu_time=$(grep -E "Time: " $cpu_out | tail -n1 | awk '{print $2}')
  gpu_time=$(grep -E "Time: " $gpu_out | tail -n1 | awk '{print $2}')
  grep '^Vertex ' $cpu_out | awk -F": " '{print $2}' > $cpu_out.dist
  grep '^Vertex ' $gpu_out | awk -F": " '{print $2}' > $gpu_out.dist
  n_cpu=$(wc -l < $cpu_out.dist); n_gpu=$(wc -l < $gpu_out.dist)
  if [ $n_cpu -gt $n_gpu ]; then seq $n_cpu | xargs -I{} echo INF >> $gpu_out.dist; fi
  if [ $n_gpu -gt $n_cpu ]; then seq $n_gpu | xargs -I{} echo INF >> $cpu_out.dist; fi
  diffs=$(paste $cpu_out.dist $gpu_out.dist | awk '{if($1!=$2) c++} END{print c+0}')
  printf "%-20s | %-12s | %-12s | %-8s\n" "$alg" "$cpu_time" "$gpu_time" "$diffs"
  rm -f $cpu_out $gpu_out $cpu_out.dist $gpu_out.dist
done
