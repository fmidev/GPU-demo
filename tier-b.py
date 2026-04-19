from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from numcodecs import blosc,Blosc
import numpy as np
import cupy as cp
import cupyx

import threading

from time import perf_counter
from functools import wraps

ARRAY_SHAPE = (67, 65, 1069, 949)
CHUNK_SHAPE = (24, 65, 200, 200)
ARRAY_ATTRS = {
    "grid_mapping": "lambert",
    "coordinates": "a b latitude longitude",
    "_ARRAY_DIMENSIONS": ["time", "hybrid", "y", "x"],
}

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
def compute(i,j,k,slot):
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
    RH.get(out=h_out_buf[slot],blocking=False)


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
threads: list[threading.Thread | None] = [None, None]
slot_chunk: list[tuple[int, int, int] | None] = [None, None]
count = 0

for i in range(3):
    for j in range(6):
        for k in range(5):
            slot = count % 2
            if threads[slot] is not None:
                threads[slot].join()
                threads[slot] = None

            with streams[slot]:
                compute(i,j,k,slot)
            slot_chunk[slot] = (i, j, k)

            if count > 0:
                prev_slot = 1 - slot
                streams[prev_slot].synchronize()
                pi, pj, pk = slot_chunk[prev_slot]
                threads[prev_slot] = threading.Thread(
                    target=write_blosc_array,
                    args=(f"out.zarr/tier_b/{pi}.0.{pj}.{pk}", h_out_buf[prev_slot], compressor),
                )
                threads[prev_slot].start()

            count += 1

if count > 0:
    final_slot = (count - 1) % 2
    streams[final_slot].synchronize()
    if threads[final_slot] is not None:
        threads[final_slot].join()
    fi, fj, fk = slot_chunk[final_slot]
    write_blosc_array(f"out.zarr/tier_b/{fi}.0.{fj}.{fk}", h_out_buf[final_slot], compressor)

for thread in threads:
    if thread is not None:
        thread.join()

write_zarr_metadata(
    "out.zarr/tier_b",
    shape=ARRAY_SHAPE,
    chunks=CHUNK_SHAPE,
    compressor_config=compressor,
)
