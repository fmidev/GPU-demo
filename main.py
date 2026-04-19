import asyncio
import logging
from time import perf_counter

import cupy as cp
import cupyx
import numpy as np

from blosc_async import read_blosc_array, write_blosc_array
from metadata import write_zarr_array_metadata

logger = logging.getLogger(__name__)


# Create memory pools for faster host/device transfers.
pinned_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)

async_pool = cp.cuda.MemoryAsyncPool()
cp.cuda.set_allocator(async_pool.malloc)

# Shared Blosc config used for all output chunks.
compressor = {
    "id": "blosc",
    "cname": "lz4",
    "clevel": 5,
    "shuffle": 1,
    "blocksize": 0,
}

CF_VERSION = "CF-1.8"
ARRAY_SHAPE = (67, 65, 1069, 949)
CHUNK_SHAPE = (24, 65, 200, 200)
ARRAY_ATTRS = {
    "grid_mapping": "lambert",
    "coordinates": "a b latitude longitude",
    "_ARRAY_DIMENSIONS": ["time", "hybrid", "y", "x"],
}


async def load_ab():
    """Load hybrid-pressure coefficients from disk."""
    a_buf = np.empty((65,), dtype=np.float64)
    b_buf = np.empty((65,), dtype=np.float64)

    await asyncio.gather(
        read_blosc_array(
            "../data/dataset.zarr/a/0",
            a_buf,
            dtype=np.float64,
            shape=(65,),
            chunk_id="0",
            task_tag="load_coefficients",
        ),
        read_blosc_array(
            "../data/dataset.zarr/b/0",
            b_buf,
            dtype=np.float64,
            shape=(65,),
            chunk_id="0",
            task_tag="load_coefficients",
        ),
    )

    a = a_buf.astype(np.float32, copy=False)
    b = b_buf.astype(np.float32, copy=False)
    return a, b


async def compute_rh_gpu_async(t_h, q_h, ps_h, stream, out_h, *, task_tag: str):
    """Launch RH work and await stream completion."""
    start_event, done_event = _launch_rh_gpu(t_h, q_h, ps_h, out_h, stream)
    await asyncio.to_thread(done_event.synchronize)
    elapsed_ms = cp.cuda.get_elapsed_time(start_event, done_event)
    logger.info("task=%s kernel=rh elapsed_ms=%.3f", task_tag, elapsed_ms)
    return out_h


def _launch_rh_gpu(
    t_h: np.ndarray,
    q_h: np.ndarray,
    ps_h: np.ndarray,
    out_h: np.ndarray,
    stream: cp.cuda.Stream,
) -> tuple[cp.cuda.Event, cp.cuda.Event]:
    """Compute relative humidity on GPU and stage async host copy."""
    with stream:
        start_event = cp.cuda.Event()
        start_event.record(stream)
        t = cp.asarray(t_h, blocking=False)
        q = cp.asarray(q_h, blocking=False)
        ps = cp.asarray(ps_h, blocking=False)

        p = (
            a[None, :, None, None]
            + b[None, :, None, None] * ps.squeeze(axis=1)[:, None, :, :]
        ) / 100
        T = t - 273.15

        E = cp.where(
            T > -5,
            6.107 * 10 ** (7.5 * T / (237 + T)),
            6.107 * 10 ** (9.5 * T / (265.5 + T)),
        )

        RH = 100 * (p * q) / (0.622 * E) * (p - E) / (p - (q * p) / 0.622)

        RH.get(out=out_h, blocking=False)
        done_event = cp.cuda.Event()
        done_event.record(stream)
        return start_event, done_event


async def compute_rho_gpu_async(t_h, ps_h, stream, out_rho, *, task_tag: str):
    """Launch density work and await stream completion."""
    start_event, done_event = _launch_rho_gpu(t_h, ps_h, out_rho, stream)
    await asyncio.to_thread(done_event.synchronize)
    elapsed_ms = cp.cuda.get_elapsed_time(start_event, done_event)
    logger.info("task=%s kernel=rho elapsed_ms=%.3f", task_tag, elapsed_ms)
    return out_rho


def _launch_rho_gpu(
    t_h: np.ndarray,
    ps_h: np.ndarray,
    out_rho: np.ndarray,
    stream: cp.cuda.Stream,
) -> tuple[cp.cuda.Event, cp.cuda.Event]:
    """Compute density on GPU and stage async host copy."""
    with stream:
        start_event = cp.cuda.Event()
        start_event.record(stream)
        t = cp.asarray(t_h, blocking=False)
        ps = cp.asarray(ps_h, blocking=False)

        P = (
            a[None, :, None, None]
            + b[None, :, None, None] * ps.squeeze(axis=1)[:, None, :, :]
        )
        RHO = P / (287 * t)
        RHO.get(out=out_rho, blocking=False)

        done_event = cp.cuda.Event()
        done_event.record(stream)
        return start_event, done_event


async def process_one(
    i: int,
    j: int,
    k: int,
    compressor: dict,
    sem: asyncio.Semaphore,
) -> int:
    """Read one chunk triplet, compute RH/RHO, then write outputs."""
    chunk_id = f"{i}.0.{j}.{k}"
    task_tag = f"chunk:{chunk_id}"
    async with sem:
        t_h = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        q_h = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        ps_h = cupyx.empty_pinned((24, 1, 200, 200), dtype=np.float32)
        out_rh = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        out_rho = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        stream1 = cp.cuda.Stream(non_blocking=True)
        stream2 = cp.cuda.Stream(non_blocking=True)

        await asyncio.gather(
            read_blosc_array(
                f"../data/dataset.zarr/t/{chunk_id}",
                t_h,
                dtype=np.float32,
                shape=(24, 65, 200, 200),
                chunk_id=chunk_id,
                task_tag=f"{task_tag}:read_t",
            ),
            read_blosc_array(
                f"../data/dataset.zarr/q/{chunk_id}",
                q_h,
                dtype=np.float32,
                shape=(24, 65, 200, 200),
                chunk_id=chunk_id,
                task_tag=f"{task_tag}:read_q",
            ),
            read_blosc_array(
                f"../data/dataset.zarr/ps/{chunk_id}",
                ps_h,
                dtype=np.float32,
                shape=(24, 1, 200, 200),
                chunk_id=chunk_id,
                task_tag=f"{task_tag}:read_ps",
            ),
        )

        out_rh, out_rho = await asyncio.gather(
            compute_rh_gpu_async(t_h, q_h, ps_h, stream1, out_rh, task_tag=task_tag),
            compute_rho_gpu_async(t_h, ps_h, stream2, out_rho, task_tag=task_tag),
        )

        await asyncio.gather(
            write_blosc_array(
                f"out.zarr/rh/{chunk_id}",
                out_rh,
                compressor,
                chunk_id=chunk_id,
                task_tag=f"{task_tag}:write_rh",
            ),
            write_blosc_array(
                f"out.zarr/rho/{chunk_id}",
                out_rho,
                compressor,
                chunk_id=chunk_id,
                task_tag=f"{task_tag}:write_rho",
            ),
        )


async def main(compressor: dict) -> None:
    """Schedule chunk processing with bounded concurrency."""
    sem = asyncio.Semaphore(4)

    tasks = [
        asyncio.create_task(process_one(i, j, k, compressor, sem))
        for i in range(3)
        for j in range(6)
        for k in range(5)
    ]

    await asyncio.gather(*tasks)
    await asyncio.gather(
        asyncio.to_thread(
            write_zarr_array_metadata,
            "out.zarr/rh",
            shape=ARRAY_SHAPE,
            chunks=CHUNK_SHAPE,
            dtype=np.float32,
            compressor_config=compressor,
            attrs=ARRAY_ATTRS,
        ),
        asyncio.to_thread(
            write_zarr_array_metadata,
            "out.zarr/rho",
            shape=ARRAY_SHAPE,
            chunks=CHUNK_SHAPE,
            dtype=np.float32,
            compressor_config=compressor,
            attrs=ARRAY_ATTRS,
        ),
    )


if __name__ == "__main__":
    logger = logging.getLogger("main")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Load coefficients once, then process all chunk tasks.
    t = perf_counter()
    a, b = asyncio.run(load_ab())
    a = cp.asarray(a)
    b = cp.asarray(b)

    asyncio.run(main(compressor))
    logger.info("task=run_total elapsed_s=%.6f", perf_counter() - t)
