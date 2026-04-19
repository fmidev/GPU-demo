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
5. Writes `.zarray` and `.zattrs` metadata files in `ds.zarr/rh/` and
   `ds.zarr/rho/` after chunk processing completes.

## Annotated tier examples

The repository also includes four progressively different RH pipelines with
inline code annotations that explain what each stage is doing and why:

- `tier-a.py`: low-level multi-buffer/stream pipeline with threaded writes.
- `tier-b.py`: simplified ping/pong double-buffer variant.
- `tier-c.py`: compact ping/pong workflow driven by a buffer generator.
- `tier-d.py`: high-level xarray+dask map-blocks pipeline with GPU kernels.

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

Expected output includes INFO logs with timings for Blosc I/O and GPU tasks,
including file paths and chunk ids, plus total runtime, for example:

```text
2026-01-01 12:00:00,000 INFO blosc_async task=chunk:0.0.0.0:read_t op=read_blosc path=... chunk_id=0.0.0.0 ... elapsed_s=...
2026-01-01 12:00:00,001 INFO __main__ task=chunk:0.0.0.0 kernel=rh elapsed_ms=...
2026-01-01 12:00:12,340 INFO __main__ task=run_total elapsed_s=12.340000
```

## Data paths

The current example expects input data under `../data/dataset.zarr/` and writes
results to `ds.zarr/` in the working directory.
