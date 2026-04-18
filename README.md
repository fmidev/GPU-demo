# GPU-demo

A demo project that reads Blosc-compressed chunks, performs GPU computation
with [CuPy](https://cupy.dev/), and writes compressed results asynchronously.

## What the code does

`main.py` orchestrates an async pipeline:

1. Loads hybrid-pressure coefficients (`a`, `b`) from
   `../data/dataset.zarr/a/0` and `../data/dataset.zarr/b/0`.
2. Reads temperature (`t`), humidity (`q`), and surface pressure (`ps`) chunks
   using `read_blosc_array` from `blosc_async.py`.
3. Launches two GPU computations on separate CUDA streams:
   - relative humidity (`rh`)
   - air density (`rho`)
4. Copies results back to pinned host memory and writes compressed output chunks
   to `ds.zarr/rh/...` and `ds.zarr/rho/...`.

## Requirements

* Python ≥ 3.10
* NVIDIA GPU with CUDA 12.x drivers
* `numpy`
* `cupy-cuda12x`
* `numcodecs`
* `blosc`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

From the repository root:

```bash
python main.py
```

Expected output includes total runtime, for example:

```text
first run took 12.34s
```

## Data paths

The current example expects input data under `../data/dataset.zarr/` and writes
results to `ds.zarr/` in the working directory.
