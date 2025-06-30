1. install Relate from `https://myersgroup.github.io/relate`

2. install tree sequence conversion tool:
```bash
git clone https://github.com/leospeidel/relate_lib.git
cd relate_lib && mkdir -p build && cd build
cmake .. && make
```

3. simulation example,
```bash
python date-island-model.py \
  --relate-dir /path/to/relate/release \
  --relate-lib-dir /path/to/relate_lib \
  --output-dir test-run \
  <other args>
```

4. plot mutational density through time,
```bash
python util/plot-time-windowed-segsites.py \
  --tree-sequence test-run/relate.trees \
  --true-tree-sequence test-run/true.trees \
  --output-path test-run/plots/time-windowed-segsites-relate.png \
  --title "Relate MCMC"

python util/plot-time-windowed-segsites.py \
  --tree-sequence test-run/ep.trees \
  --true-tree-sequence test-run/true.trees \
  --output-path test-run/plots/time-windowed-segsites-ep.png \
  --title "Relate EP"
```
