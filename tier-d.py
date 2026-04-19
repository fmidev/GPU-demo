import xarray as xr
import numpy as np
import cupy as cp
import dask.array as da
import dask
from numcodecs import Blosc
import zarr

dask.config.set({"array.chunk-size": 249600000})

ds = xr.open_zarr("../data/dataset.zarr")
a = ds["a"].astype("float32").data
b = ds["b"].astype("float32").data
t = ds["t"].data
q = ds["q"].data
ps = ds["ps"].data

def relative_humidity_gpu(t_block, q_block, ps_block, a_coeff, b_coeff):
    t_gpu = cp.asarray(t_block)
    q_gpu = cp.asarray(q_block)
    ps_gpu = cp.asarray(ps_block)
    a_gpu = cp.asarray(a_coeff)
    b_gpu = cp.asarray(b_coeff)
    ps_surface_gpu = ps_gpu[:, 0, :, :] if ps_gpu.ndim == 4 else ps_gpu

    p_gpu = (
        a_gpu[None, :, None, None]
        + b_gpu[None, :, None, None] * ps_surface_gpu[:, None, :, :]
    ) / 100
    t_celsius_gpu = t_gpu - 273.15
    svp_gpu = cp.where(
        t_celsius_gpu > -5,
        6.107 * 10 ** (7.5 * t_celsius_gpu / (237 + t_celsius_gpu)),
        6.107 * 10 ** (9.5 * t_celsius_gpu / (265.5 + t_celsius_gpu)),
    )
    rh_gpu = 100 * (p_gpu * q_gpu) / (0.622 * svp_gpu) * (
        (p_gpu - svp_gpu) / (p_gpu - (q_gpu * p_gpu) / 0.622)
    )
    return cp.asnumpy(rh_gpu)

z = da.map_blocks(
        relative_humidity_gpu,
        t,
        q,
        ps,
        a,
        b,
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
