"""
GPU Demo: Chunked host-to-device streaming with CuPy CUDA streams.

Data is held in chunked arrays on host memory (NumPy). Each chunk is
streamed to the GPU using a dedicated CuPy CUDA stream, the square root
of every element is computed on the device, and the result is streamed
back to the host.
"""

import numpy as np
import cupy as cp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOTAL_ELEMENTS = 1_000_000   # total number of float32 values
CHUNK_SIZE = 100_000          # elements per chunk
N_STREAMS = 4                 # number of concurrent CUDA streams


def make_host_chunks(total: int, chunk_size: int) -> list[np.ndarray]:
    """Allocate and initialise source data as a list of host (NumPy) chunks.

    Each chunk is a 1-D float32 array filled with random values in [0, 1).
    """
    rng = np.random.default_rng(seed=42)
    source = rng.random(total, dtype=np.float32)
    return [source[i : i + chunk_size] for i in range(0, total, chunk_size)]


def process_chunks(
    chunks: list[np.ndarray],
    n_streams: int = N_STREAMS,
) -> list[np.ndarray]:
    """Stream each chunk to the GPU, compute sqrt, stream result back.

    Parameters
    ----------
    chunks:
        List of host (NumPy) arrays to process.
    n_streams:
        Number of CUDA streams to use for concurrent transfers and kernel
        launches.

    Returns
    -------
    List of host (NumPy) arrays containing sqrt(chunk) for every input chunk.
    """
    # Pre-allocate CUDA streams (round-robin across chunks).
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]

    results: list[np.ndarray | None] = [None] * len(chunks)

    # --- enqueue work on streams -------------------------------------------
    # We keep device arrays alive until their stream has finished, so we
    # store them alongside the stream they were launched on.
    pending: list[tuple[int, cp.cuda.Stream, cp.ndarray]] = []

    for idx, host_chunk in enumerate(chunks):
        stream = streams[idx % n_streams]

        with stream:
            # Asynchronously transfer host chunk → device
            device_chunk = cp.asarray(host_chunk)
            # Compute sqrt on the device (kernel launches on this stream)
            device_result = cp.sqrt(device_chunk)

        pending.append((idx, stream, device_result))

    # --- collect results ---------------------------------------------------
    for idx, stream, device_result in pending:
        stream.synchronize()
        # Transfer result device → host
        results[idx] = cp.asnumpy(device_result)

    return results  # type: ignore[return-value]


def verify(
    chunks: list[np.ndarray],
    results: list[np.ndarray],
    *,
    rtol: float = 1e-5,
) -> bool:
    """Check that every result matches numpy's sqrt on the original chunk."""
    for i, (chunk, result) in enumerate(zip(chunks, results)):
        expected = np.sqrt(chunk)
        if not np.allclose(result, expected, rtol=rtol):
            print(f"  Chunk {i}: MISMATCH")
            return False
    return True


def main() -> None:
    print("=== GPU Demo: chunked sqrt via CUDA streams ===\n")
    print(f"  Total elements : {TOTAL_ELEMENTS:,}")
    print(f"  Chunk size     : {CHUNK_SIZE:,}")
    print(f"  Number of chunks: {-(-TOTAL_ELEMENTS // CHUNK_SIZE)}")  # ceil div
    print(f"  CUDA streams   : {N_STREAMS}")
    print()

    # 1. Build chunked host data
    print("Initialising host data with random numbers …")
    chunks = make_host_chunks(TOTAL_ELEMENTS, CHUNK_SIZE)
    print(f"  Created {len(chunks)} chunks, each of shape {chunks[0].shape}\n")

    # 2. Stream chunks to GPU, compute sqrt, stream back
    print("Streaming chunks to GPU, computing sqrt, streaming results back …")
    results = process_chunks(chunks, n_streams=N_STREAMS)
    print(f"  Processed {len(results)} chunks\n")

    # 3. Verify correctness
    print("Verifying results against NumPy reference …")
    ok = verify(chunks, results)
    status = "PASSED ✓" if ok else "FAILED ✗"
    print(f"  Verification: {status}\n")

    # 4. Show a small sample
    sample_chunk = results[0][:8]
    print("Sample (first 8 sqrt values from chunk 0):")
    print(" ", sample_chunk)


if __name__ == "__main__":
    main()
