# Report


## Environment

### Setup DGL

You can install `dgl` from conda with CUDA support

```shell
conda install -c dglteam dgl-cuda11.3
```

For CPU only version, run:

```shell
conda install -c dglteam dgl
```

On macOS, you can install dgl From source

```shell
mkdir build
cd build
#cmake -DUSE_CUDA=ON ..
cmake -DUSE_OPENMP=off -DCMAKE_C_FLAGS='-DXBYAK_DONT_USE_MAP_JIT' -DCMAKE_CXX_FLAGS='-DXBYAK_DONT_USE_MAP_JIT' -DUSE_AVX=OFF -DUSE_LIBXSMM=OFF ..
make -j4
```

Then, install python bindings

```shell
cd ../python
python setup.py install
```
