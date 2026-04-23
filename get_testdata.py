import argparse
import os

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_path",
        nargs="?",
        default="input/input.zarr",
        help="Output Zarr path (default: input/input.zarr)",
    )
    args = parser.parse_args()

    storage_options = {
        "anon": True,
        "client_kwargs": {"endpoint_url": "https://lake.fmi.fi"},
        "config_kwargs": {
            "response_checksum_validation": "when_required",
            "request_checksum_calculation": "when_required",
        },
    }

    ds = xr.open_zarr("s3://dask-datasets/testdata/meps/fc202604210300-rechunk.zarr/", storage_options=storage_options)

    ds_out = ds[["t", "q", "ps", "lambert"]]

    parent_dir = os.path.dirname(args.output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    ds_out.to_zarr(args.output_path, zarr_format=2, mode="w")


if __name__ == "__main__":
    main()
