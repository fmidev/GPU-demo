# GPU-demo

A demo project that shows how to stream data chunk-wise from host memory to a
CUDA device and back using [CuPy](https://cupy.dev/) CUDA streams.

## What the demo does

1. **Allocates** a large 1-D float32 array on the host (NumPy) and splits it
   into equal-sized chunks.
2. **Initialises** every source chunk with random values (NumPy random
   generator).
3. **Streams** each chunk asynchronously to the GPU using a dedicated
   `cupy.cuda.Stream` (*async GPU*).
4. **Computes** the element-wise square root on the device (`cupy.sqrt`).
5. **Streams** the result back to the host.
6. **Naive GPU**: copies each chunk to the device in a plain for-loop (no
   streams), computes sqrt, and copies the result back before moving to the
   next chunk.
7. **CPU baseline**: computes sqrt with NumPy for comparison.
8. **Verifies** the results of both GPU implementations against the NumPy
   reference.

The number of concurrent CUDA streams is configurable; by default four streams
are used so that transfers and kernel launches for different chunks can overlap.

## Requirements

* Python ≥ 3.10
* NVIDIA GPU with CUDA 12.x drivers
* `numpy`
* `cupy-cuda12x`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python demo.py
```

Example output:

```
=== GPU Demo: chunked sqrt via CUDA streams ===

  Total elements : 1,000,000
  Chunk size     : 100,000
  Number of chunks: 10
  CUDA streams   : 4

Initialising host data with random numbers …
  Created 10 chunks, each of shape (100000,)

GPU async  – streaming chunks to device, computing sqrt, streaming back …
  Processed 10 chunks

GPU naive  – copying chunks in a loop (no streams), computing sqrt …
  Processed 10 chunks

CPU  – computing sqrt with NumPy …
  Processed 10 chunks

Verifying GPU async results against CPU reference …
  Verification: PASSED ✓

Verifying GPU naive results against CPU reference …
  Verification: PASSED ✓

────────────────────────────────────────────────────────
                     Timing summary
────────────────────────────────────────────────────────
  GPU async wall time (upload + kernel + download):   12.34 ms
  GPU async kernel-only time (CUDA events, aggregated):  2.10 ms
  GPU naive wall time (loop copy + kernel + copy back):  18.50 ms
  CPU wall time (NumPy sqrt, all chunks):               45.67 ms
────────────────────────────────────────────────────────
  Speedup  GPU async wall vs CPU wall:                  3.70×
  Speedup  GPU naive wall vs CPU wall:                  2.47×
────────────────────────────────────────────────────────

Sample (first 8 sqrt values from chunk 0):
  [0.77459663 0.9958897  0.31053504 …]
```

## Configuration

Edit the constants at the top of `demo.py` to change the workload:

| Constant          | Default     | Description                        |
|-------------------|-------------|------------------------------------|
| `TOTAL_ELEMENTS`  | 1 000 000   | Total number of float32 values     |
| `CHUNK_SIZE`      | 100 000     | Elements per chunk                 |
| `N_STREAMS`       | 4           | Number of concurrent CUDA streams  |
