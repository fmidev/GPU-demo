from __future__ import annotations

import json
import argparse
import blosc
from pathlib import Path
from typing import Union

from numcodecs import Blosc
import numpy as np
import cupy as cp
import cupyx

import threading

ARRAY_SHAPE = (67, 65, 1069, 949)
CHUNK_SHAPE = (24, 65, 200, 200)
ARRAY_ATTRS = {
    "grid_mapping": "lambert",
    "coordinates": "a b latitude longitude",
    "_ARRAY_DIMENSIONS": ["time", "hybrid", "y", "x"],
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zarr", default="../data/dataset.zarr")
    parser.add_argument("--output-zarr", default="out.zarr")
    return parser.parse_args()

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
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array)

    codec = Blosc(
        cname=compressor_config["cname"],
        clevel=compressor_config["clevel"],
        shuffle=compressor_config["shuffle"],
        blocksize=compressor_config["blocksize"],
    )

    compressed = codec.encode(array)
    path.write_bytes(compressed)

def write_zarr_metadata(
    array_dir: Union[str, Path],
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    compressor_config: dict,
) -> None:
    array_dir = Path(array_dir)
    array_dir.mkdir(parents=True, exist_ok=True)

    zarray = {
        "chunks": list(chunks),
        "compressor": compressor_config,
        "dtype": np.dtype(np.float32).newbyteorder("<").str,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": list(shape),
        "zarr_format": 2,
    }

    (array_dir / ".zarray").write_text(json.dumps(zarray, indent=2) + "\n", encoding="utf-8")
    (array_dir / ".zattrs").write_text(
        json.dumps(ARRAY_ATTRS, indent=2) + "\n", encoding="utf-8"
    )

def read_blosc_array(
    file_path: Union[str, Path],
    dst: np.ndarray,
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
        The same destination array (`dst`) after in-place decompression.

    Raises
    ------
    ValueError
        If the decompressed byte size does not match dtype * shape.
    """
    file_path = Path(file_path)

    dtype = np.dtype(dtype)
    expected_nbytes = int(np.prod(shape)) * dtype.itemsize

    if order not in ("C", "F"):
        raise ValueError("`order` must be 'C' or 'F'.")
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

    compressed = file_path.read_bytes()
    nbytes, _cbytes, _blocksize = blosc.get_cbuffer_sizes(compressed)
    if nbytes != expected_nbytes:
        raise ValueError(
            f"Decompressed data size mismatch: got {nbytes} bytes, "
            f"expected {expected_nbytes} bytes for shape={shape}, dtype={dtype}."
        )
    blosc.decompress_ptr(compressed, dst.ctypes.data)
    return dst

def compute(i, j, k, buf):
    t = cp.asarray(read_blosc_array(input_zarr / "t" / f"{i}.0.{j}.{k}", dst=t_in_buf, dtype=np.float32, shape=(24, 65, 200, 200)), blocking=False)
    q = cp.asarray(read_blosc_array(input_zarr / "q" / f"{i}.0.{j}.{k}", dst=q_in_buf, dtype=np.float32, shape=(24, 65, 200, 200)), blocking=False)
    ps = cp.asarray(read_blosc_array(input_zarr / "ps" / f"{i}.0.{j}.{k}", dst=ps_in_buf, dtype=np.float32, shape=(24, 1, 200, 200)), blocking=False)
    ps = cp.squeeze(ps, axis=1)

    p = (a[None, :, None, None] + b[None, :, None, None] * ps[:, None, :, :]) / 100
    T = t - 273.15

    E = cp.where(
        T > -5,
        6.107 * 10 ** (7.5 * T / (237 + T)),
        6.107 * 10 ** (9.5 * T / (265.5 + T)),
        )

    RH = 100 * (p * q) / (0.622 * E) * (p - E) / (p - (q*p) / 0.622)
    RH.get(out=buf,blocking=False)


##############
#   START   #
#############
args = parse_args()
input_zarr = Path(args.input_zarr)
output_zarr = Path(args.output_zarr)

compressor = {
            "id": "blosc",
            "cname": "lz4",
            "clevel": 5,
            "shuffle": 1,
            "blocksize": 0,
            }

NUM_BUFFERS = 4

buffers = [
    cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
    for _ in range(NUM_BUFFERS)
]

a_in_buf = np.empty((65,), dtype=np.float64)
b_in_buf = np.empty((65,), dtype=np.float64)
t_in_buf = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
q_in_buf = cupyx.empty_pinned((24, 65, 200, 200), dtype=np.float32)
ps_in_buf = cupyx.empty_pinned((24, 1, 200, 200), dtype=np.float32)
streams = [cp.cuda.Stream(non_blocking=True) for _ in range(NUM_BUFFERS)]
threads: list[threading.Thread | None] = [None for _ in range(NUM_BUFFERS)]

a = cp.asarray(read_blosc_array(input_zarr / "a" / "0", dst=a_in_buf, dtype=np.float64, shape=(65,)).astype(np.float32))
b = cp.asarray(read_blosc_array(input_zarr / "b" / "0", dst=b_in_buf, dtype=np.float64, shape=(65,)).astype(np.float32))


count = 0
prev_ijk: tuple[int, int, int] | None = None

for i in range(3):
    for j in range(6):
        for k in range(5):
            cur = count % NUM_BUFFERS
            io_thread = threads[cur]
            if io_thread is not None:
                io_thread.join()
                    
            with streams[cur]:
                compute(i, j, k, buffers[cur])

            if prev_ijk is not None:
                streams[prev].synchronize()
                pi, pj, pk = prev_ijk
                threads[prev] = threading.Thread(
                    target=write_blosc_array,
                    args=(output_zarr / "tier_a" / f"{pi}.0.{pj}.{pk}", buffers[prev], compressor),
                )
                threads[prev].start()

            prev_ijk = (i, j, k)
            prev = cur
            count += 1

if prev_ijk is not None:
    streams[prev].synchronize()
    io_thread = threads[prev]
    if io_thread is not None:
        io_thread.join()
    pi, pj, pk = prev_ijk
    write_blosc_array(output_zarr / "tier_a" / f"{pi}.0.{pj}.{pk}", buffers[prev], compressor)

for io_thread in threads:
    if io_thread is not None:
        io_thread.join()

write_zarr_metadata(
    output_zarr / "tier_a",
    shape=ARRAY_SHAPE,
    chunks=CHUNK_SHAPE,
    compressor_config=compressor,
)
