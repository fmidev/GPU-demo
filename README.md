# GPU-demo

A demo project that reads Blosc-compressed chunks, performs GPU computation
with [CuPy](https://cupy.dev/), and writes compressed results asynchronously.

## What the code does

`main.py` orchestrates an async pipeline:

1. Loads hybrid-pressure coefficients (`a`, `b`) from
   `<input-zarr>/a/0` and `<input-zarr>/b/0`.
2. Reads temperature (`t`), humidity (`q`), and surface pressure (`ps`) chunks
   using `read_blosc_array` from `blosc_async.py`.
3. Launches two GPU computations on separate CUDA streams:
   - relative humidity (`rh`)
   - air density (`rho`)
4. Copies results back to pinned host memory and writes compressed output chunks
   to `<output-zarr>/rh/...` and `<output-zarr>/rho/...`.
5. Writes `.zarray` and `.zattrs` metadata files in `<output-zarr>/rh/` and
   `<output-zarr>/rho/` after chunk processing completes.

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
python main.py --input-zarr ../data/dataset.zarr --output-zarr out.zarr
```

### Retrieve test data from S3 and run GPU calculations

Use the helper script to download a local zarr test dataset:

```bash
python get_testdata.py input/input.zarr
```

Then run the GPU pipeline with that input:

```bash
python main.py --input-zarr input/input.zarr --output-zarr out.zarr
```

Expected output includes INFO logs with timings for Blosc I/O and GPU tasks,
including file paths and chunk ids, plus total runtime, for example:

```text
2026-01-01 12:00:00,000 INFO blosc_async task=chunk:0.0.0.0:read_t op=read_blosc path=... chunk_id=0.0.0.0 ... elapsed_s=...
2026-01-01 12:00:00,001 INFO __main__ task=chunk:0.0.0.0 kernel=rh elapsed_ms=...
2026-01-01 12:00:12,340 INFO __main__ task=run_total elapsed_s=12.340000
```

## Data paths

`main.py`, `tier-a.py`, `tier-b.py`, `tier-c.py`, and `tier-d.py` all accept:

- `--input-zarr` (default: `../data/dataset.zarr`)
- `--output-zarr` (default: `out.zarr`)

The scripts keep variable directories under the zarr trunks (for example
`<input-zarr>/t/...`, `<output-zarr>/rh/...`, `<output-zarr>/tier_a/...`).
