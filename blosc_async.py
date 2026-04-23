from __future__ import annotations

import os
import asyncio
import logging
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Union

import blosc
import numpy as np
from numcodecs import Blosc

logger = logging.getLogger(__name__)

async def write_blosc_array(
    path: Union[str, Path],
    array: np.ndarray,
    compressor_config: dict,
    *,
    chunk_id: str | None = None,
    task_tag: str = "blosc_write",
) -> None:
    """Write a NumPy array as a Blosc-compressed byte stream."""
    path = Path(path)
    started = perf_counter()

    def _write() -> None:
        contiguous = np.ascontiguousarray(array)
        codec = Blosc(
            cname=compressor_config["cname"],
            clevel=compressor_config["clevel"],
            shuffle=compressor_config["shuffle"],
            blocksize=compressor_config["blocksize"],
        )
        compressed = codec.encode(contiguous)

        path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            tmp.write(compressed)
            tmp_name = tmp.name

        os.replace(tmp_name, path)

    await asyncio.to_thread(_write)
    elapsed = perf_counter() - started
    logger.info(
        "task=%s op=write_blosc path=%s elapsed_s=%.6f",
        task_tag,
        path,
        elapsed,
    )


async def read_blosc_array(
    file_path: Union[str, Path],
    dst: np.ndarray,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    order: str = "C",
    chunk_id: str | None = None,
    task_tag: str = "blosc_read",
) -> None:
    """Read a Blosc file into an existing NumPy destination array."""
    file_path = Path(file_path)
    dtype = np.dtype(dtype)
    expected_nbytes = int(np.prod(shape)) * dtype.itemsize
    started = perf_counter()

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
        # Validate decompressed size before writing into destination memory.
        compressed = file_path.read_bytes()
        nbytes, _cbytes, _blocksize = blosc.get_cbuffer_sizes(compressed)
        if nbytes != expected_nbytes:
            raise ValueError(
                f"Decompressed data size mismatch: got {nbytes} bytes, "
                f"expected {expected_nbytes} bytes for shape={shape}, dtype={dtype}."
            )
        blosc.decompress_ptr(compressed, dst.ctypes.data)

    await asyncio.to_thread(_read)
    elapsed = perf_counter() - started
    logger.info(
        "task=%s op=read_blosc path=%s elapsed_s=%.6f",
        task_tag,
        file_path,
        elapsed,
    )
