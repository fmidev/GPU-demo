import xarray as xr

ds = xr.open_zarr("../data/dataset.zarr")

p = (ds.a.astype("float32") + ds.b.astype("float32") * ds.ps.squeeze()) / 100
T = ds.t - 273.15
E = xr.where(
    T > -5,
    6.107 * 10 ** (7.5 * T / (237 + T)),
    6.107 * 10 ** (9.5 * T / (265.5 + T)),
)

ds["RH"] = 100 * (p * ds.q) / (0.622 * E) * (p - E) / (p - (ds.q*p) / 0.622)

ds.RH.to_dataset().to_zarr("out.zarr",zarr_format=2,mode="w")
