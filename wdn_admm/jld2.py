"""Minimal reader for Julia ``JLD2`` files.

``JLD2`` is an HDF5-based container, so :mod:`h5py` can open the files
directly.  What it *cannot* do on its own is undo the Julia-specific
encodings that the original code relies on:

* Julia stores arrays column-major, so a Julia ``(nn, nt)`` matrix shows up
  in ``h5py`` with shape ``(nt, nn)`` and has to be transposed.
* ``SparseMatrixCSC`` is written as a compound type holding object
  references to the ``colptr`` / ``rowval`` / ``nzval`` vectors, all with
  1-based indices.
* ``Union{Nothing, T}`` element types (which is what ``v_loc`` and
  ``y_loc`` end up as) are written as a ``(mask, tN)`` compound.

This module hides all of that behind :func:`load_jld2`, which returns a
plain ``dict`` of NumPy / SciPy objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp

__all__ = ["load_jld2", "read_jld2_value"]

# JLD2 writes the reconstructed-type table under this group; it is metadata,
# not payload, so it never shows up in the returned dictionary.
_TYPES_GROUP = "_types"


def _is_sparse_csc(dtype: np.dtype) -> bool:
    names = dtype.names
    return names is not None and set(names) >= {"m", "n", "colptr", "rowval", "nzval"}


def _is_union(dtype: np.dtype) -> bool:
    """Detect the ``Union{Nothing, T}`` compound layout ``(mask, t1, t2, ...)``."""
    names = dtype.names
    if names is None:
        return False
    return names[0] == "mask" and all(n.startswith("t") for n in names[1:])


def _deref(f: h5py.File, value: Any) -> Any:
    """Follow an HDF5 object reference (JLD2 uses them for nested arrays)."""
    if isinstance(value, h5py.Reference):
        return f[value][()]
    return value


def _decode_sparse(f: h5py.File, record: np.void) -> sp.csc_matrix:
    m = int(record["m"])
    n = int(record["n"])
    colptr = np.asarray(_deref(f, record["colptr"]), dtype=np.int64) - 1
    rowval = np.asarray(_deref(f, record["rowval"]), dtype=np.int64) - 1
    nzval = np.asarray(_deref(f, record["nzval"]))
    return sp.csc_matrix((nzval.astype(float), rowval, colptr), shape=(m, n))


def _decode_union(record: np.ndarray) -> np.ndarray:
    """Unwrap ``Union{Nothing, T}`` values into a plain array.

    JLD2 stores the active branch of the union in the ``mask`` field.  For the
    problem data shipped with this repository every entry is populated, so we
    simply take the last (non-``Nothing``) payload field.  Entries whose mask
    marks them as ``Nothing`` are returned as NaN.
    """
    payload_fields = [n for n in record.dtype.names if n != "mask"]
    values = np.asarray(record[payload_fields[-1]])
    mask = np.asarray(record["mask"])
    # mask == 0 is the `Nothing` branch in the JLD2 union encoding.
    if np.any(mask == 0):
        values = values.astype(float)
        values[mask == 0] = np.nan
    return values


def read_jld2_value(f: h5py.File, name: str) -> Any:
    """Read and decode a single top-level entry of an open JLD2 file."""
    dataset = f[name]
    if isinstance(dataset, h5py.Group):
        return {key: read_jld2_value(dataset, key) for key in dataset}

    raw = dataset[()]
    dtype = dataset.dtype

    if _is_sparse_csc(dtype):
        return _decode_sparse(f, raw)
    if _is_union(dtype):
        return _decode_union(raw)
    if dtype == object or h5py.check_ref_dtype(dtype):
        # Vector-of-vectors (`Vector{Vector{T}}` / `Array{Any}`).
        flat = [np.asarray(_deref(f, item)) for item in np.ravel(raw)]
        try:
            return np.stack(flat).T if flat and flat[0].ndim else np.asarray(flat)
        except ValueError:
            return flat

    array = np.asarray(raw)
    if array.ndim >= 2:
        # Undo Julia's column-major ordering.
        array = array.T
    if array.ndim == 0:
        return array.item()
    return array


def load_jld2(path: str | Path) -> dict[str, Any]:
    """Load every top-level variable of ``path`` into a dictionary.

    Matrices are transposed back to their Julia orientation, sparse matrices
    become :class:`scipy.sparse.csc_matrix`, and scalars become Python
    numbers.  Index vectors keep their original 1-based values; converting
    them is the job of :mod:`wdn_admm.data`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JLD2 file not found: {path}")

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        for name in f:
            if name == _TYPES_GROUP:
                continue
            out[name] = read_jld2_value(f, name)
    return out
