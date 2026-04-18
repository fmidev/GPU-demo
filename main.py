import numpy as np
import cupy as cp
import cupyx
import threading
import asyncio

from time import perf_counter
from blosc_async import *
from mempool import *



# create memory pool for pinned host memory
pinned_pool = cp.cuda.PinnedMemoryPool()
cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)

async_pool = cp.cuda.MemoryAsyncPool()
cp.cuda.set_allocator(async_pool.malloc)

compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
            }

async def load_ab():
    a_buf = np.empty((65,), dtype=np.float64)
    b_buf = np.empty((65,), dtype=np.float64)

    await asyncio.gather(
        read_blosc_array("../data/dataset.zarr/a/0", a_buf, dtype=np.float64, shape=(65,)),
        read_blosc_array("../data/dataset.zarr/b/0", b_buf, dtype=np.float64, shape=(65,)),
    )

    a = a_buf.astype(np.float32, copy=False)
    b = b_buf.astype(np.float32, copy=False)
    return a, b

async def compute_rh_gpu_async(t_h, q_h, ps_h, stream, out_h):
    event = _launch_rh_gpu(t_h, q_h, ps_h, out_h, stream)
    await asyncio.to_thread(event.synchronize)

    return out_h

def _launch_rh_gpu(
    t_h: np.ndarray,
    q_h: np.ndarray,
    ps_h: np.ndarray,
    out_h: np.ndarray,
    stream: cp.cuda.Stream,
) -> cp.cuda.Event:

    with stream:
        t = cp.asarray(t_h,blocking=False)
        q = cp.asarray(q_h,blocking=False)
        ps = cp.asarray(ps_h,blocking=False)
 
        p = (a[None, :, None, None] + b[None, :, None, None] * ps.squeeze(axis=1)[:, None, :, :]) / 100
        T = t - 273.15

        E = cp.where(
            T > -5,
            6.107 * 10 ** (7.5 * T / (237 + T)),
            6.107 * 10 ** (9.5 * T / (265.5 + T)),
            )

        RH = 100 * (p * q) / (0.622 * E) * (p - E) / (p - (q*p) / 0.622)

        RH.get(out=out_h, blocking=False)
        event = cp.cuda.Event()
        event.record(stream)

        return event

async def compute_rho_gpu_async(t_h, ps_h, stream, out_rho):
    event = _launch_rho_gpu(t_h, ps_h, out_rho, stream)
    await asyncio.to_thread(event.synchronize)

    return out_rho

def _launch_rho_gpu(
    t_h: np.ndarray,
    ps_h: np.ndarray,
    out_rho: np.ndarray,
    stream: cp.cuda.Stream,
) -> cp.cuda.Event:

    with stream:
        t = cp.asarray(t_h,blocking=False)
        ps = cp.asarray(ps_h,blocking=False)

        P = (a[None, :, None, None] + b[None, :, None, None] * ps.squeeze(axis=1)[:, None, :, :])

        RHO = P / (287*t)
        RHO.get(out=out_rho, blocking=False)

        event = cp.cuda.Event()
        event.record(stream)

        return event

async def process_one(i: int, j: int, k: int, compressor: dict, sem: asyncio.Semaphore) -> int:
    async with sem:
        t_h = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        q_h = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        ps_h = cupyx.empty_pinned((24, 1, 200, 200), dtype=np.float32)
        out_rh = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        out_rho = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
        stream1 = cp.cuda.Stream(non_blocking=True)
        stream2 = cp.cuda.Stream(non_blocking=True)

        await asyncio.gather(read_blosc_array(
            f"../data/dataset.zarr/t/{i}.0.{j}.{k}",
            t_h,
            dtype=np.float32,
            shape=(24, 65, 200, 200),
        ),
        read_blosc_array(
            f"../data/dataset.zarr/q/{i}.0.{j}.{k}",
            q_h,
            dtype=np.float32,
            shape=(24, 65, 200, 200),
        ),
        read_blosc_array(
            f"../data/dataset.zarr/ps/{i}.0.{j}.{k}",
            ps_h,
            dtype=np.float32,
            shape=(24, 1, 200, 200),
        )
        )

        out_rh, out_rho = await asyncio.gather(
                compute_rh_gpu_async(t_h, q_h, ps_h, stream1, out_rh),
                compute_rho_gpu_async(t_h, ps_h, stream2, out_rho)
                )

        await asyncio.gather(write_blosc_array(
            f"ds.zarr/rh/{i}.0.{j}.{k}",
            out_rh,
            compressor,
        ),
        write_blosc_array(
            f"ds.zarr/rho/{i}.0.{j}.{k}",
            out_rho,
            compressor,
        )
        )

async def main(compressor: dict) -> None:
    sem = asyncio.Semaphore(4)

    tasks = [
        asyncio.create_task(process_one(i, j, k, compressor, sem))
        for i in range(3)
        for j in range(6)
        for k in range(5)
    ]

    await asyncio.gather(*tasks)

# start timing
t = perf_counter()

# Load a,b first
a, b = asyncio.run(load_ab())
a = cp.asarray(a)
b = cp.asarray(b)

# run the job
asyncio.run(main(compressor))
print(f"first run took {perf_counter() -t}s")

