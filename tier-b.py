from __future__ import annotations

from pathlib import Path
from typing import Union

from numcodecs import blosc,Blosc
import numpy as np
import cupy as cp
import cupyx

import threading

from time import perf_counter
from functools import wraps

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

@time_it
def write_blosc_array(
    path: Union[str, Path],
    array: np.ndarray,
    #*,
    compressor_config: dict,
) -> None:
    """
    Write a NumPy array to a file as a numcodecs Blosc-compressed byte stream.

    Parameters
    ----------
    path
        Output file path.
    array
        NumPy array to compress.
    compressor_config
        Dict like:
        {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
        }
    """
    path = Path(path)
    array = np.ascontiguousarray(array)

    codec = Blosc(
        cname=compressor_config["cname"],
        clevel=compressor_config["clevel"],
        shuffle=compressor_config["shuffle"],
        blocksize=compressor_config["blocksize"],
    )

    compressed = codec.encode(array)
    path.write_bytes(compressed)

@time_it
def read_blosc_array(
    file_path: Union[str, Path],
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    order: str = "C",
) -> np.ndarray:
    """
    Read a Blosc-compressed binary file into a NumPy array.

    Parameters
    ----------
    file_path
        Path to the Blosc-compressed file.
    dtype
        NumPy dtype of the original array, e.g. np.float32.
    shape
        Shape of the original array, e.g. (1000, 256).
    order
        Memory order used when reshaping. Usually "C".

    Returns
    -------
    np.ndarray
        Decompressed NumPy array.

    Raises
    ------
    ValueError
        If the decompressed byte size does not match dtype * shape.
    """
    file_path = Path(file_path)

    compressed = file_path.read_bytes()
    decompressed = blosc.decompress(compressed)

    dtype = np.dtype(dtype)
    expected_nbytes = int(np.prod(shape)) * dtype.itemsize

    if len(decompressed) != expected_nbytes:
        raise ValueError(
            f"Decompressed data size mismatch: got {len(decompressed)} bytes, "
            f"expected {expected_nbytes} bytes for shape={shape}, dtype={dtype}."
        )

    return np.frombuffer(decompressed, dtype=dtype).reshape(shape, order=order)

@time_it
def compute(i,j,k):
    t = cp.asarray(read_blosc_array(f"../data/dataset.zarr/t/{i}.0.{j}.{k}",dtype=np.float32,shape=(24,65,200,200)))
    q = cp.asarray(read_blosc_array(f"../data/dataset.zarr/q/{i}.0.{j}.{k}",dtype=np.float32,shape=(24,65,200,200)))
    ps = cp.asarray(read_blosc_array(f"../data/dataset.zarr/ps/{i}.0.{j}.{k}",dtype=np.float32,shape=(24,1,200,200)))
    ps = cp.squeeze(ps, axis=1)

    p = (a[None, :, None, None] + b[None, :, None, None] * ps[:, None, :, :]) / 100
    T = t - 273.15

    E = cp.where(
        T > -5,
        6.107 * 10 ** (7.5 * T / (237 + T)),
        6.107 * 10 ** (9.5 * T / (265.5 + T)),
        )

    RH = 100 * (p * q) / (0.622 * E) * (p - E) / (p - (q*p) / 0.622)
    RH.get(out=h_out_buf[k%2],blocking=False)


a=cp.asarray(read_blosc_array(f"../data/dataset.zarr/a/0",dtype=np.float64,shape=(65)).astype(np.float32))
b=cp.asarray(read_blosc_array(f"../data/dataset.zarr/b/0",dtype=np.float64,shape=(65)).astype(np.float32))

compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
            }

streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
h_out_buf = [cupyx.empty_pinned((24,65,200,200), dtype=np.float32) for _ in range(2)]
threads = [threading.Thread() for _ in range(2)]
prev = 0
count = 0

for i in range(3):
    for j in range(6):
        for k in range(5):
            with streams[k%2]:
                compute(i,j,k)

            streams[prev].synchronize()
            try:
                threads[prev].join()
            except:
                pass
            if count > 0:
                pi, pj, pk = prev_ijk
                threads[prev] = threading.Thread(target=write_blosc_array, args=(f"ds.zarr/rh/{pi}.0.{pj}.{pk}",h_out_buf[prev],compressor,))
                threads[prev].start()

            prev = k%2
            prev_ijk = i,j,k
            count += 1

streams[prev].synchronize()
write_blosc_array(f"ds.zarr/rh/{i}.0.{j}.{k}",h_out_buf[prev],compressor)

