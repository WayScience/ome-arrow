"""Tests for TensorView DLPack export and iteration."""

import pathlib

import numpy as np
import pyarrow.compute as pc
import pytest

from ome_arrow import OMEArrow
from ome_arrow.export import to_ome_parquet


def _from_dlpack_capsule(capsule: object) -> np.ndarray:
    if hasattr(capsule, "__dlpack__"):
        return np.from_dlpack(capsule)

    class _Wrapper:
        def __init__(self, cap: object) -> None:
            self._cap = cap

        def __dlpack__(self, stream: object | None = None) -> object:
            _ = stream
            return self._cap

        def __dlpack_device__(self) -> tuple[int, int]:
            return (1, 0)

    return np.from_dlpack(_Wrapper(capsule))


def test_tensor_view_layout_and_values(example_correct_data: dict) -> None:
    """Validate TensorView layout defaults and permutation behavior."""
    oa = OMEArrow(example_correct_data)

    view = oa.tensor_view(t=0, z=0)
    assert view.layout == "CHW"
    arr = view.to_numpy()
    assert arr.shape == (2, 3, 4)

    expected = np.stack(
        [
            np.array(example_correct_data["planes"][0]["pixels"]).reshape(3, 4),
            np.array(example_correct_data["planes"][1]["pixels"]).reshape(3, 4),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(arr, expected)

    view_hwc = oa.tensor_view(t=0, z=0, layout="HWC")
    arr_hwc = view_hwc.to_numpy(contiguous=False)
    assert arr_hwc.shape == (3, 4, 2)
    assert arr_hwc[0, 0, 0] == expected[0, 0, 0]
    assert arr_hwc[0, 0, 1] == expected[1, 0, 0]
    assert not arr_hwc.flags["C_CONTIGUOUS"]

    arr_hwc_contig = view_hwc.to_numpy(contiguous=True)
    assert arr_hwc_contig.flags["C_CONTIGUOUS"]


def test_dlpack_roundtrip_torch(example_correct_data: dict) -> None:
    """Round-trip DLPack export/import through torch on CPU."""
    torch = pytest.importorskip("torch")

    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0)
    dlpack = view.to_dlpack(contiguous=False, mode="numpy")
    tensor = torch.utils.dlpack.from_dlpack(dlpack)

    expected = np.stack(
        [
            np.array(example_correct_data["planes"][0]["pixels"]).reshape(3, 4),
            np.array(example_correct_data["planes"][1]["pixels"]).reshape(3, 4),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(tensor.cpu().numpy(), expected)
    assert tensor.dtype == torch.uint16

    # Pointer equality here relies on TensorView reusing the same cached
    # materialization for both to_dlpack(mode="numpy") and to_numpy().
    arr = view.to_numpy(contiguous=False)
    assert tensor.data_ptr() == arr.__array_interface__["data"][0]


def test_dlpack_roundtrip_jax(example_correct_data: dict) -> None:
    """Round-trip DLPack export/import through JAX on CPU."""
    jax = pytest.importorskip("jax")

    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0)
    dlpack = view.to_dlpack(contiguous=True, mode="numpy")
    arr = jax.dlpack.from_dlpack(dlpack)

    expected = np.stack(
        [
            np.array(example_correct_data["planes"][0]["pixels"]).reshape(3, 4),
            np.array(example_correct_data["planes"][1]["pixels"]).reshape(3, 4),
        ],
        axis=0,
    )
    np.testing.assert_array_equal(np.array(arr), expected)


def test_dlpack_invalid_device(example_correct_data: dict) -> None:
    """Raise a clear error for unsupported devices."""
    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0)
    with pytest.raises(ValueError, match="Unsupported device"):
        view.to_dlpack(device="tpu")


def test_dlpack_arrow_mode_single_plane(example_correct_data: dict) -> None:
    """Export a single plane in arrow mode as a flat 1D buffer."""
    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0, c=0)

    dlpack = view.to_dlpack(mode="arrow")
    arr = _from_dlpack_capsule(dlpack)
    expected = np.array(example_correct_data["planes"][0]["pixels"])
    np.testing.assert_array_equal(arr, expected)


def test_layout_drop_non_singleton_errors() -> None:
    """Reject layout drops when the omitted axis is non-singleton."""
    arr = np.zeros((2, 1, 1, 2, 2), dtype=np.uint16)
    oa = OMEArrow(arr)
    view = oa.tensor_view(layout="CHW")
    with pytest.raises(ValueError, match="drops non-singleton"):
        view.to_numpy()


def test_iter_dlpack_batches() -> None:
    """Yield expected batch counts and reconstruct original data."""
    arr = np.arange(3 * 1 * 1 * 2 * 2, dtype=np.uint16).reshape(3, 1, 1, 2, 2)
    oa = OMEArrow(arr)
    view = oa.tensor_view()

    batches = list(view.iter_dlpack(batch_size=2, shuffle=False, mode="numpy"))
    assert len(batches) == 2

    batch_arrays = [_from_dlpack_capsule(b) for b in batches]
    assert batch_arrays[0].shape[0] == 2
    assert batch_arrays[1].shape[0] == 1

    recon = np.concatenate(batch_arrays, axis=0)
    expected = view.to_numpy(contiguous=True)
    np.testing.assert_array_equal(recon, expected)


def test_iter_dlpack_shuffle_deterministic() -> None:
    """Keep shuffle order deterministic with a fixed seed."""
    arr = np.arange(4 * 1 * 1 * 2 * 2, dtype=np.uint16).reshape(4, 1, 1, 2, 2)
    oa = OMEArrow(arr)
    view = oa.tensor_view()

    caps_a = list(view.iter_dlpack(batch_size=1, shuffle=True, seed=7, mode="numpy"))
    caps_b = list(view.iter_dlpack(batch_size=1, shuffle=True, seed=7, mode="numpy"))

    vals_a = [_from_dlpack_capsule(c)[0, 0, 0, 0, 0] for c in caps_a]
    vals_b = [_from_dlpack_capsule(c)[0, 0, 0, 0, 0] for c in caps_b]
    assert vals_a == vals_b


def test_iter_dlpack_tiles(example_correct_data: dict) -> None:
    """Yield tiled DLPack payloads with expected shapes."""
    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0)

    caps = list(view.iter_dlpack(tiles=(2, 2), mode="numpy"))
    assert len(caps) == 4

    tile = _from_dlpack_capsule(caps[0])
    assert tile.shape == (2, 2, 2)


def test_arrow_mode_zero_copy_parquet(
    tmp_path: pathlib.Path, example_correct_data: dict
) -> None:
    """Best-effort pointer check for torch zero-copy from parquet."""
    torch = pytest.importorskip("torch")

    out = tmp_path / "example.parquet"
    to_ome_parquet(example_correct_data, out_path=str(out), column_name="ome_arrow")

    oa = OMEArrow(str(out))
    view = oa.tensor_view(t=0, z=0, c=0)
    capsule = view.to_dlpack(mode="arrow")

    struct_arr = oa._struct_array
    assert struct_arr is not None
    planes = struct_arr.field("planes")[0].values
    mask = pc.and_(
        pc.equal(planes.field("t"), 0),
        pc.and_(pc.equal(planes.field("z"), 0), pc.equal(planes.field("c"), 0)),
    )
    selected = pc.filter(planes, mask)
    pixels_list = selected.field("pixels")[0]
    values = pixels_list.values

    offset_bytes = values.offset * (values.type.bit_width // 8)
    ptr_arrow = values.buffers()[1].address + offset_bytes
    ptr_torch = torch.utils.dlpack.from_dlpack(capsule).data_ptr()
    if ptr_torch != ptr_arrow:
        pytest.skip("Torch did not preserve zero-copy for pyarrow DLPack.")


def test_arrow_mode_zero_copy_parquet_jax(
    tmp_path: pathlib.Path, example_correct_data: dict
) -> None:
    """Best-effort pointer check for JAX zero-copy from parquet."""
    jax = pytest.importorskip("jax")

    out = tmp_path / "example.parquet"
    to_ome_parquet(example_correct_data, out_path=str(out), column_name="ome_arrow")

    oa = OMEArrow(str(out))
    view = oa.tensor_view(t=0, z=0, c=0)
    capsule = view.to_dlpack(mode="arrow")

    arr = jax.dlpack.from_dlpack(capsule)
    try:
        device = arr.device()
        assert device.platform == "cpu"
    except Exception:
        pytest.skip("Unable to read JAX device platform.")

    struct_arr = oa._struct_array
    assert struct_arr is not None
    planes = struct_arr.field("planes")[0].values
    mask = pc.and_(
        pc.equal(planes.field("t"), 0),
        pc.and_(pc.equal(planes.field("z"), 0), pc.equal(planes.field("c"), 0)),
    )
    selected = pc.filter(planes, mask)
    pixels_list = selected.field("pixels")[0]
    values = pixels_list.values

    offset_bytes = values.offset * (values.type.bit_width // 8)
    ptr_arrow = values.buffers()[1].address + offset_bytes
    ptr_jax = _jax_buffer_ptr(arr)
    if ptr_jax != ptr_arrow:
        pytest.skip("JAX did not preserve zero-copy for pyarrow DLPack.")


def _jax_buffer_ptr(arr: object) -> int:
    if hasattr(arr, "device_buffer"):
        buf = arr.device_buffer
    elif hasattr(arr, "device_buffers"):
        buffers = arr.device_buffers
        if not buffers:
            raise AssertionError("JAX array has no device buffers.")
        buf = buffers[0]
    elif hasattr(arr, "addressable_data"):
        buf = arr.addressable_data(0)
    else:
        raise AssertionError("Unable to access JAX device buffer.")

    if hasattr(buf, "unsafe_buffer_pointer"):
        return int(buf.unsafe_buffer_pointer())
    inner = getattr(buf, "buffer", None)
    if inner is not None and hasattr(inner, "unsafe_buffer_pointer"):
        return int(inner.unsafe_buffer_pointer())
    raise AssertionError("Unable to access JAX buffer pointer.")
