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

def compute(i,j,k,buf):
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
    RH.get(out=buf,blocking=True)


a=cp.asarray(read_blosc_array(f"../data/dataset.zarr/a/0",dtype=np.float64,shape=(65)).astype(np.float32))
b=cp.asarray(read_blosc_array(f"../data/dataset.zarr/b/0",dtype=np.float64,shape=(65)).astype(np.float32))

compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
            }

streams = [cp.cuda.Stream(non_blocking=True) for _ in range(5)]
ping = {"buffer" : cupyx.empty_pinned((24,65,200,200), dtype=np.float32), "stream" : cp.cuda.Stream(non_blocking=True)}
pong = {"buffer" : cupyx.empty_pinned((24,65,200,200), dtype=np.float32), "stream" : cp.cuda.Stream(non_blocking=True)}

pingpong = (ping,pong)

io_thread = threading.Thread()

prev = 0
count = 0

for i in range(3):
    for j in range(6):
        for k in range(5):
            s = pingpong[count%2]["stream"]
            with s:
                compute(i,j,k,pingpong[count%2]["buffer"])
            
            try:
                io_thread.join()
            except:
                pass

            io_thread = threading.Thread(target=write_blosc_array, args=(f"out.zarr/tier_c/{i}.0.{j}.{k}",pingpong[count%2]["buffer"],compressor,))
            io_thread.start()

            count += 1
