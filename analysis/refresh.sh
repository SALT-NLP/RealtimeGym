#!/bin/bash
# Pull finished CHTC results and rebuild every figure and replay in one go.
#   ./analysis/refresh.sh            # full refresh
#   NO_REPLAYS=1 ./analysis/refresh.sh   # figures only (faster)
#   SKIP_DOWNLOAD=1 ./analysis/refresh.sh   # rebuild from what is already local
set -e
cd "$(dirname "$0")/.."
REPO=$(pwd)
RESULTS=$REPO/rtg_results
FIGS=$REPO/rtg_figures
PY=${PYTHON:-python3}

if ! $PY -c "import matplotlib, numpy, pandas, pygame" 2>/dev/null; then
  echo "Missing packages. Run:  $PY -m pip install matplotlib numpy pandas pygame"
  exit 1
fi

mkdir -p "$RESULTS" "$FIGS/replays"
echo "== downloading finished results (password + Duo if no SSH session is open)"
[ -n "$SKIP_DOWNLOAD" ] || scp -q 'nbag@ap2001.chtc.wisc.edu:/staging/n/nbag/rtg_results/*.tar.gz' "$RESULTS/"

echo "== extracting"
cd "$RESULTS"
for f in *.tar.gz; do tar -xzf "$f"; done
cd "$REPO"

echo "== cells"
find "$RESULTS/logs" -name args.log | while read -r f; do
  d=$(dirname "$f")
  tag=$(head -1 "$(dirname "$d")/model.txt" 2>/dev/null | awk '{print tolower($2)}')
  # started: from the run folder name (..._YYYYMMDD_HHMMSS, UTC on the exec node)
  # took: longest seed's total_time, i.e. the wall-clock length of the eval
  started=$(basename "$d" | sed -E 's/.*_([0-9]{8})_([0-9]{2})([0-9]{2})[0-9]{2}$/\1 \2:\3/')
  took=$(grep '^seed:' "$f" | sed 's/.*total_time: \([0-9.]*\).*/\1/' | sort -n | tail -1 \
         | awk '{printf "%.1fh", $1/3600}')
  printf "  %-12s %-10s %-9s %s %6s  seeds=%s  started=%s  took=%-6s\n" "${tag:-qwen3-8b}" \
    "$(grep -m1 '^game:' "$f" | awk '{print $2}')" \
    "$(grep -m1 '^mode:' "$f" | awk '{print $2}')" \
    "$(grep -m1 '^cognitive_load:' "$f" | awk '{print $2}')" \
    "$(grep -m1 '^time_pressure:' "$f" | awk '{print $2}')" \
    "$(grep -c '^seed:' "$f")" "$started" "$took"
done | sort

echo "== figures"
$PY analysis/plot_baseline.py --logs_root "$RESULTS/logs" --out "$FIGS" | grep -E "===|Figure|Summary"

if [ -z "$NO_REPLAYS" ]; then
  echo "== replays (new runs only)"
  find "$RESULTS/logs" -name args.log | sort | while read -r f; do
    d=$(dirname "$f")
    name=$(basename "$d" | sed 's/_2026[0-9_]*$//')
    runtag=$(basename "$(dirname "$d")")
    out="$FIGS/replays/${name}__${runtag}.html"
    if [ ! -f "$out" ] || [ "$f" -nt "$out" ]; then
      $PY analysis/replay_viewer.py --run_dir "$d" --out "$out" | grep -E "DIVERGED|Viewer" | sed "s|$REPO/||"
    fi
  done
fi
echo "== done: open $FIGS"
