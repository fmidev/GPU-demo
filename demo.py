"""
GPU Demo: Chunked host-to-device streaming with CuPy CUDA streams.

Data is held in chunked arrays on CUDA pinned (page-locked) host memory.
Each chunk is streamed to the GPU using a dedicated CuPy CUDA stream, the
square root of every element is computed on the device, and the result is
streamed back to the host.

Pinned memory allows the CUDA DMA engine to bypass the OS bounce buffer,
improving host↔device transfer throughput.

Timing statistics are printed at the end to compare GPU (full pipeline:
host→device + kernel + device→host) against a pure-CPU NumPy baseline.
"""

import time

import numpy as np
import cupy as cp


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOTAL_ELEMENTS = 1_000_000   # total number of float32 values
CHUNK_SIZE = 100_000          # elements per chunk
N_STREAMS = 4                 # number of concurrent CUDA streams


def pinned_array(array: np.ndarray) -> np.ndarray:
    """Return a copy of *array* backed by CUDA pinned (page-locked) memory.

    Pinned memory allows the CUDA DMA engine to transfer data between host and
    device without an intermediate bounce buffer, which can significantly
    improve host↔device bandwidth.

    Memory lifecycle: ``np.frombuffer`` retains a reference to the underlying
    ``PinnedMemoryPointer`` (``mem``) via Python's buffer protocol.  The pinned
    allocation is therefore kept alive for as long as the returned array is
    alive, and is freed automatically by CuPy's memory pool once the array is
    garbage-collected.
    """
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    src = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    src[...] = array
    return src


def make_host_chunks(total: int, chunk_size: int) -> list[np.ndarray]:
    """Allocate and initialise source data as a list of pinned host chunks.

    Each chunk is a 1-D float32 array filled with random values in [0, 1)
    and is backed by CUDA pinned (page-locked) memory so that host↔device
    DMA transfers bypass any intermediate bounce buffer.
    """
    rng = np.random.default_rng(seed=42)
    source = rng.random(total, dtype=np.float32)
    return [
        pinned_array(source[i : i + chunk_size])
        for i in range(0, total, chunk_size)
    ]


def process_chunks(
    chunks: list[np.ndarray],
    n_streams: int = N_STREAMS,
) -> tuple[list[np.ndarray], float, float]:
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
    results:
        List of host (NumPy) arrays containing sqrt(chunk) for every input
        chunk.
    kernel_ms:
        Aggregate GPU kernel time in milliseconds measured with CUDA events
        (excludes host↔device transfer time).
    wall_s:
        Wall-clock elapsed seconds for the entire pipeline (transfers +
        kernels + synchronisation).
    """
    # Pre-allocate CUDA streams and event pairs (round-robin across chunks).
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(n_streams)]
    # One pair of CUDA events per chunk so each kernel interval is timed
    # independently.  Events are allocated up-front to avoid per-iteration
    # overhead inside the hot loop.
    ev_starts = [cp.cuda.Event() for _ in range(len(chunks))]
    ev_ends = [cp.cuda.Event() for _ in range(len(chunks))]

    results: list[np.ndarray | None] = [None] * len(chunks)

    # --- enqueue work on streams -------------------------------------------
    # We keep device arrays alive until their stream has finished, so we
    # store them alongside the stream they were launched on.
    # CUDA events bracket each kernel for precise on-device timing.
    pending: list[tuple[int, cp.cuda.Stream, cp.ndarray,
                        cp.cuda.Event, cp.cuda.Event]] = []

    wall_start = time.perf_counter()

    for idx, host_chunk in enumerate(chunks):
        stream = streams[idx % n_streams]
        ev_start = ev_starts[idx]
        ev_end = ev_ends[idx]

        with stream:
            # Asynchronously transfer host chunk → device.
            device_chunk = cp.asarray(host_chunk)
            # ev_start is enqueued on the stream *after* the H2D transfer, so
            # get_elapsed_time(ev_start, ev_end) captures only kernel time.
            ev_start.record(stream)
            # Compute sqrt on the device (kernel launches on this stream)
            device_result = cp.sqrt(device_chunk)
            ev_end.record(stream)

        pending.append((idx, stream, device_result, ev_start, ev_end))

    # --- collect results ---------------------------------------------------
    kernel_ms = 0.0
    for idx, stream, device_result, ev_start, ev_end in pending:
        stream.synchronize()
        # stream.synchronize() guarantees all operations on the stream are
        # complete, including event recording, so no extra event sync needed.
        # Transfer result device → host
        results[idx] = cp.asnumpy(device_result)
        # Accumulate kernel time (get_elapsed_time returns milliseconds)
        kernel_ms += cp.cuda.get_elapsed_time(ev_start, ev_end)

    wall_s = time.perf_counter() - wall_start

    return results, kernel_ms, wall_s  # type: ignore[return-value]


def process_chunks_cpu(chunks: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """Compute sqrt on the CPU for every chunk using NumPy.

    Parameters
    ----------
    chunks:
        List of host (NumPy) arrays to process.

    Returns
    -------
    results:
        List of host (NumPy) arrays containing sqrt(chunk).
    wall_s:
        Wall-clock elapsed seconds.
    """
    start = time.perf_counter()
    results = [np.sqrt(chunk) for chunk in chunks]
    wall_s = time.perf_counter() - start
    return results, wall_s


def verify(
    gpu_results: list[np.ndarray],
    cpu_results: list[np.ndarray],
    *,
    rtol: float = 1e-5,
) -> bool:
    """Check that every GPU result matches the corresponding CPU result."""
    for i, (gpu, cpu) in enumerate(zip(gpu_results, cpu_results)):
        if not np.allclose(gpu, cpu, rtol=rtol):
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

    # 2. GPU pipeline: stream chunks to device, compute sqrt, stream back
    print("GPU  – streaming chunks to device, computing sqrt, streaming back …")
    gpu_results, kernel_ms, gpu_wall_s = process_chunks(chunks, n_streams=N_STREAMS)
    print(f"  Processed {len(gpu_results)} chunks\n")

    # 3. CPU baseline: compute sqrt with NumPy
    print("CPU  – computing sqrt with NumPy …")
    cpu_results, cpu_wall_s = process_chunks_cpu(chunks)
    print(f"  Processed {len(cpu_results)} chunks\n")

    # 4. Verify GPU results against CPU reference
    print("Verifying GPU results against CPU reference …")
    ok = verify(gpu_results, cpu_results)
    status = "PASSED ✓" if ok else "FAILED ✗"
    print(f"  Verification: {status}\n")

    # 5. Timing summary
    _eps = 1e-9  # guard against division by zero for very fast GPU runs
    speedup = cpu_wall_s / max(gpu_wall_s, _eps)
    print("─" * 52)
    print(f"{'Timing summary':^52}")
    print("─" * 52)
    print(f"  {'GPU wall time (upload + kernel + download):':<44} {gpu_wall_s * 1e3:>6.2f} ms")
    print(f"  {'GPU kernel-only time (CUDA events, aggregated):':<44} {kernel_ms:>6.2f} ms")
    print(f"  {'CPU wall time (NumPy sqrt, all chunks):':<44} {cpu_wall_s * 1e3:>6.2f} ms")
    print("─" * 52)
    print(f"  {'Speedup  GPU wall vs CPU wall:':<44} {speedup:>6.2f}×")
    print("─" * 52)
    print()

    # 6. Show a small sample
    sample = gpu_results[0][:8]
    print("Sample (first 8 sqrt values from chunk 0):")
    print(" ", sample)


if __name__ == "__main__":
    main()
