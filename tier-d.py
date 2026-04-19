import xarray as xr
import numpy as np
import cupy as cp
import dask.array as da
import dask
from numcodecs import Blosc
import zarr

dask.config.set({"array.chunk-size": 249600000})

ds = xr.open_zarr("../data/dataset.zarr")
x = ds["t"].data
y = ds["q"].data

def add_gpu(a,b):
    a_gpu = cp.asarray(a)
    b_gpu = cp.asarray(b)
    ab_gpu = a_gpu + b_gpu
    return cp.asnumpy(ab_gpu)

z = da.map_blocks(
        add_gpu,
        x,
        y,
        dtype=np.float32
)

compressor = Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)
target = zarr.open(
    "out.zarr",
    mode="w",
    shape=z.shape,
    chunks=z.chunksize,
    dtype=z.dtype,
    compressor=compressor,
    zarr_format=2,
)

z.to_zarr(target, overwrite=True)
