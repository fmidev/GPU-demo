from __future__ import annotations

from pathlib import Path
from typing import Union

from numcodecs import Blosc
import blosc
import numpy as np
from time import perf_counter
from functools import wraps
import asyncio

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        result = func(*args, **kwargs)
        end = perf_counter()
        print(f"{func.__name__} started {start}, ended {end}, took {end - start:.6f} seconds")
        return result
    return wrapper

@time_it
async def write_blosc_array(
    path: Union[str, Path],
    array: np.ndarray,
    compressor_config: dict,
) -> None:
    """
    Write a NumPy array to a file as a numcodecs Blosc-compressed byte stream.
    """
    path = Path(path)

    def _write() -> None:
        contiguous = np.ascontiguousarray(array)

        codec = Blosc(
            cname=compressor_config["cname"],
            clevel=compressor_config["clevel"],
            shuffle=compressor_config["shuffle"],
            blocksize=compressor_config["blocksize"],
        )

        compressed = codec.encode(contiguous)
        path.write_bytes(compressed)

    await asyncio.to_thread(_write)

@time_it
async def read_blosc_array(
    file_path: Union[str, Path],
    dst: np.ndarray,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    order: str = "C",
) -> None:
    """
    Read a Blosc-compressed binary file into an existing NumPy array.
    """
    file_path = Path(file_path)
    dtype = np.dtype(dtype)
    expected_nbytes = int(np.prod(shape)) * dtype.itemsize

    if dst.dtype != dtype:
        raise ValueError(f"`dst.dtype` is {dst.dtype}, expected {dtype}.")

    if dst.shape != shape:
        raise ValueError(f"`dst.shape` is {dst.shape}, expected {shape}.")

    if dst.nbytes != expected_nbytes:
        raise ValueError(
            f"`dst` has {dst.nbytes} bytes, expected {expected_nbytes} bytes "
            f"for shape={shape}, dtype={dtype}."
        )

    if order == "C" and not dst.flags.c_contiguous:
        raise ValueError("`dst` must be C-contiguous.")
    if order == "F" and not dst.flags.f_contiguous:
        raise ValueError("`dst` must be F-contiguous.")
    if order not in ("C", "F"):
        raise ValueError("`order` must be 'C' or 'F'.")

    def _read() -> None:
        compressed = file_path.read_bytes()

        # Blosc header tells you the uncompressed byte size.
        nbytes, _cbytes, _blocksize = blosc.get_cbuffer_sizes(compressed)
        if nbytes != expected_nbytes:
            raise ValueError(
                f"Decompressed data size mismatch: got {nbytes} bytes, "
                f"expected {expected_nbytes} bytes for shape={shape}, dtype={dtype}."
            )

        blosc.decompress_ptr(compressed, dst.ctypes.data)

    await asyncio.to_thread(_read)
