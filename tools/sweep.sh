#!/usr/bin/env bash
# Sweeps topology, size, degree and weight spread into one csv.
set -euo pipefail

BIN=${BIN:-bin}
OUT=${OUT:-results}
GRAPHS=${GRAPHS:-graphs/sweep}
REPS=${REPS:-5}
# Sources per graph. More than one turns every row into a distribution over
# source vertices rather than a property of vertex 0.
SOURCES=${SOURCES:-5}
# RESIDENT=1 keeps the graph on the device between sources, which is the
# amortised-upload measurement rather than the one-shot one.
RESIDENT=${RESIDENT:-0}
# Extra flags handed to gen_graph.
GENFLAGS=${GENFLAGS:-}

SIZES=${SIZES:-"100000 1000000 4000000"}
DEGREES=${DEGREES:-"8 32"}
TOPOS=${TOPOS:-"uniform rmat geometric grid"}
WEIGHTS=${WEIGHTS:-"1 100 100000"}

mkdir -p "$OUT" "$GRAPHS"
stamp=$(date +%Y%m%d_%H%M%S)
# CSV= appends to an existing file, so an interrupted sweep can be finished off
# with a narrowed TOPOS/SIZES/WEIGHTS instead of being restarted from scratch.
csv=${CSV:-"$OUT/sweep_$stamp.csv"}

benchflags=(--reps "$REPS" --sources "$SOURCES")
[ "$RESIDENT" = 1 ] && benchflags+=(--resident)

for topo in $TOPOS; do
  for n in $SIZES; do
    for deg in $DEGREES; do
      for wmax in $WEIGHTS; do
        # grid, chain and star ignore the edge count, so run them once per
        # (n, wmax).
        case "$topo" in
          grid|chain|star) [ "$deg" = "8" ] || continue ;;
        esac

        m=$((n * deg / 2))
        f="$GRAPHS/${topo}_n${n}_d${deg}_w${wmax}.bin"
        tag="${topo}/n${n}/deg${deg}/w${wmax}"

        generated=0
        if [ ! -f "$f" ]; then
          "$BIN/gen_graph" --n "$n" --m "$m" --topo "$topo" \
            --wmax "$wmax" $GENFLAGS --out "$f"
          generated=1
        fi

        echo "== $tag"
        "$BIN/bench" "$f" "${benchflags[@]}" --csv "$csv" --tag "$tag"
        echo

        # The full matrix is ~16G of graphs if they all stay on disk, which is
        # more than the home quota. gen_graph is seeded, so a graph rebuilds
        # byte for byte from the loop variables; keep one at a time and drop it
        # once benched. KEEP=1 retains them, and anything already on disk when
        # the sweep started is left alone.
        if [ "$generated" = 1 ] && [ "${KEEP:-0}" != 1 ]; then rm -f "$f"; fi
      done
    done
  done
done

echo "csv: $csv"
