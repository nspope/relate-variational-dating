1. install Relate from `https://myersgroup.github.io/relate`

2. install tree sequence conversion tools:
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
  --output-dir test \
  <other args>
```
