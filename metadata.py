from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import DTypeLike


def write_zarr_array_metadata(
    array_dir: Union[str, Path],
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: DTypeLike,
    compressor_config: dict,
    attrs: dict,
) -> None:
    """Create minimal Zarr v2 metadata files for one array directory."""
    array_dir = Path(array_dir)
    array_dir.mkdir(parents=True, exist_ok=True)

    zarray = {
        "chunks": list(chunks),
        "compressor": compressor_config,
        "dtype": np.dtype(dtype).newbyteorder("<").str,
        "fill_value": None,
        "filters": None,
        "order": "C",
        "shape": list(shape),
        "zarr_format": 2,
    }

    (array_dir / ".zarray").write_text(
        json.dumps(zarray, indent=2) + "\n", encoding="utf-8"
    )
    (array_dir / ".zattrs").write_text(
        json.dumps(attrs, indent=2) + "\n", encoding="utf-8"
    )
