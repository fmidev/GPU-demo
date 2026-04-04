# GPU-demo

A demo project that shows how to stream data chunk-wise from host memory to a
CUDA device and back using [CuPy](https://cupy.dev/) CUDA streams.

## What the demo does

1. **Allocates** a large 1-D float32 array on the host (NumPy) and splits it
   into equal-sized chunks.
2. **Initialises** every source chunk with random values (NumPy random
   generator).
3. **Streams** each chunk asynchronously to the GPU using a dedicated
   `cupy.cuda.Stream`.
4. **Computes** the element-wise square root on the device (`cupy.sqrt`).
5. **Streams** the result back to the host.
6. **Verifies** the results against a NumPy reference.

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

Streaming chunks to GPU, computing sqrt, streaming results back …
  Processed 10 chunks

Verifying results against NumPy reference …
  Verification: PASSED ✓

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
