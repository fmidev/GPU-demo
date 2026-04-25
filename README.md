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

## Tier scripts (GPU implementation levels)

The tier scripts represent a spectrum of GPU optimization, from the most
optimized (tier A) to the least optimized (tier D):

| Tier | Optimization level | Key technique |
|------|--------------------|---------------|
| `tier-a.py` | **Highest** | Reads data directly into pinned (page-locked) host buffers, maximizing PCIe transfer throughput |
| `tier-b.py` | High | Triple-buffered pipeline that overlaps GPU compute and I/O across three in-flight chunks |
| `tier-c.py` | Moderate | Ping-pong (double) buffering that overlaps the write of one chunk with the compute of the next |
| `tier-d.py` | **Lowest** | High-level Dask/Xarray abstraction; GPU is used only inside `map_blocks`, with no explicit memory management or stream control |

- `tier-a.py` *(most GPU-optimized)*: Extends tier B by decompressing Blosc
  chunks directly into pinned (page-locked) memory buffers allocated with
  `cupyx.empty_pinned`.  Pinned memory enables faster host-to-device transfers
  and allows non-blocking `cp.asarray` calls, so the CPU and GPU can overlap
  work with minimal synchronisation overhead.
- `tier-b.py`: Uses three pinned output buffers and three CUDA streams to keep
  the GPU busy: while one stream computes, the previous stream's result is
  written to disk by a background thread.
- `tier-c.py`: Simpler ping-pong approach with two pinned output buffers.
  Compute and write alternate between the two buffers, providing a basic
  compute–write overlap without the extra complexity of triple buffering.
- `tier-d.py` *(least GPU-optimized)*: Delegates all scheduling and memory
  management to Dask and Xarray.  `dask.array.map_blocks` calls a Python
  function that moves each chunk to the GPU with `cp.asarray` and back with
  `cp.asnumpy`, with no pinned memory, explicit CUDA streams, or manual
  buffer reuse.

## Kernel equations

The GPU kernels compute two meteorological fields from NWP model-level data.
All input arrays are read as `float32`.

### Step 1 — Hybrid pressure

Model-level pressure `p` (hPa) is reconstructed from the hybrid vertical
coordinates and surface pressure:

```
p = (a + b · pₛ) / 100
```

| Symbol | Unit | Description |
|--------|------|-------------|
| `a`    | Pa   | Hybrid A-coefficient (pressure contribution at each model level) |
| `b`    | –    | Hybrid B-coefficient (sigma contribution, dimensionless) |
| `pₛ`   | Pa   | Surface pressure |
| `p`    | hPa  | Reconstructed model-level pressure |

### Step 2 — Temperature conversion

Air temperature is converted from Kelvin to Celsius for the saturation
vapor pressure formula:

```
T_C = T − 273.15
```

| Symbol | Unit | Description |
|--------|------|-------------|
| `T`    | K    | Air temperature (model input) |
| `T_C`  | °C   | Air temperature in Celsius |

### Step 3 — Saturation vapor pressure (Magnus formula)

The saturation vapor pressure `E` (hPa) is computed with the Magnus formula.
Two coefficient sets are used depending on whether the air is above liquid
water or ice:

```
        ⎧ 6.107 × 10^( 7.5 · T_C / (237.0 + T_C) )   if T_C > −5 °C  (over liquid water)
E(T) =  ⎨
        ⎩ 6.107 × 10^( 9.5 · T_C / (265.5 + T_C) )   if T_C ≤ −5 °C  (over ice)
```

| Symbol | Unit | Description |
|--------|------|-------------|
| `T_C`  | °C   | Air temperature in Celsius |
| `E`    | hPa  | Saturation vapor pressure |

### Step 4 — Relative humidity

Relative humidity `RH` (%) is the ratio of the mixing ratio to the
saturation mixing ratio:

```
r   = q / (1 − q)              # mixing ratio from specific humidity
r_s = 0.622 · E / (p − E)     # saturation mixing ratio

RH  = 100 · r / r_s
```

Expanding and rearranging gives the form used in the kernels:

```
RH = 100 · ((p · q) / (0.622 · E)) · ((p − E) / (p − q · p / 0.622))
```

| Symbol  | Unit   | Description |
|---------|--------|-------------|
| `q`     | kg/kg  | Specific humidity (model input) |
| `p`     | hPa    | Model-level pressure (Step 1) |
| `E`     | hPa    | Saturation vapor pressure (Step 3) |
| `0.622` | –      | Ratio of molar masses of water vapor and dry air (M_w / M_d ≈ 18.015 / 28.964) |
| `RH`    | %      | Relative humidity |

### Step 5 — Air density (main.py only)

`main.py` additionally computes air density `ρ` (kg m⁻³) from the ideal gas
law for dry air:

```
ρ = P / (R_d · T)
```

| Symbol | Unit        | Description |
|--------|-------------|-------------|
| `P`    | Pa          | Model-level pressure (= `a + b · pₛ`, **not** divided by 100) |
| `T`    | K           | Air temperature (model input) |
| `R_d`  | J kg⁻¹ K⁻¹ | Specific gas constant for dry air (287 J kg⁻¹ K⁻¹) |
| `ρ`    | kg m⁻³     | Air density |

## Data paths

`main.py`, `tier-a.py`, `tier-b.py`, `tier-c.py`, and `tier-d.py` all accept:

- `--input-zarr` (default: `../data/dataset.zarr`)
- `--output-zarr` (default: `out.zarr`)

The scripts keep variable directories under the zarr trunks (for example
`<input-zarr>/t/...`, `<output-zarr>/rh/...`, `<output-zarr>/tier_a/...`).
