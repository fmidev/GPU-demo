import xarray as xr
import numpy as np
import cupy as cp
import dask.array as da
import dask
from numcodecs import Blosc

# Let Dask choose chunks up to this byte size when auto-rechunking operations.
dask.config.set({"array.chunk-size": 249600000})

# Open the source Zarr dataset lazily (arrays remain dask-backed).
ds = xr.open_zarr("../data/dataset.zarr")
a = ds["a"].astype("float32").data
b = ds["b"].astype("float32").data
t = ds["t"].data
q = ds["q"].data
ps = ds["ps"].data

def relative_humidity_gpu(t_block, q_block, ps_block, a_coeff, b_coeff):
    # Per-block function executed by Dask: move block slices to GPU.
    t_gpu = cp.asarray(t_block)
    q_gpu = cp.asarray(q_block)
    ps_gpu = cp.asarray(ps_block)
    a_gpu = cp.asarray(a_coeff)
    b_gpu = cp.asarray(b_coeff)
    ps_surface_gpu = ps_gpu[:, 0, :, :] if ps_gpu.ndim == 4 else ps_gpu

    # Same RH equation as lower tiers, evaluated on CuPy arrays.
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

# Build a lazy Dask graph that applies the GPU function block-wise.
z = da.map_blocks(
    relative_humidity_gpu,
    t,
    q,
    ps,
    a,
    b,
    dtype=np.float32,
)

compressor = Blosc(cname="lz4", clevel=5, shuffle=1, blocksize=0)
# Carry forward metadata so output is self-describing in xarray/Zarr readers.
tier_d_attrs = {
    "standard_name": "relative_humidity",
    "long_name": "relative humidity",
    "units": "%",
}
if "grid_mapping" in ds["t"].attrs:
    tier_d_attrs["grid_mapping"] = ds["t"].attrs["grid_mapping"]

tier_d = xr.DataArray(
    z,
    dims=ds["t"].dims,
    coords=ds["t"].coords,
    name="tier_d",
    attrs=tier_d_attrs,
)

# Assumes base output dataset already exists; append only `tier_d`.
# to_zarr triggers Dask execution and writes compressed chunks to disk.
tier_d.to_dataset().to_zarr(
    "out.zarr",
    zarr_format=2,
    mode="a",
    encoding={"tier_d": {"compressor": compressor}},
)
