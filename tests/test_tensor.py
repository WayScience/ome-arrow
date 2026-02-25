"""Tests for TensorView DLPack export and iteration."""

import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import ome_arrow.core as core_module
from ome_arrow import OMEArrow
from ome_arrow.export import to_ome_parquet
from ome_arrow.meta import OME_ARROW_STRUCT
from ome_arrow.tensor import LazyTensorView, TensorView


def _from_dlpack_capsule(capsule: object) -> np.ndarray:
    if hasattr(capsule, "__dlpack__"):
        return np.from_dlpack(capsule)

    class _Wrapper:
        def __init__(self, cap: object) -> None:
            self._cap = cap

        def __dlpack__(self, stream: object | None = None) -> object:
            _ = stream
            if self._cap is None:
                raise RuntimeError("DLPack capsule has already been consumed.")
            cap = self._cap
            self._cap = None
            return cap

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

    view_yxc = oa.tensor_view(t=0, z=0, layout="YXC")
    arr_yxc = view_yxc.to_numpy(contiguous=False)
    assert arr_yxc.shape == (3, 4, 2)
    assert arr_yxc[0, 0, 0] == expected[0, 0, 0]
    assert arr_yxc[0, 0, 1] == expected[1, 0, 0]
    assert not arr_yxc.flags["C_CONTIGUOUS"]

    arr_yxc_contig = view_yxc.to_numpy(contiguous=True)
    assert arr_yxc_contig.flags["C_CONTIGUOUS"]


def test_tensor_view_chunk_policy_modes(example_correct_data: dict) -> None:
    """Control chunk handling strategy for ChunkedArray-backed inputs."""
    base = pa.array([example_correct_data], type=OME_ARROW_STRUCT)
    chunked = pa.chunked_array([base, base], type=OME_ARROW_STRUCT)

    view_auto = TensorView(chunked, chunk_policy="auto")
    assert isinstance(view_auto._data, pa.ChunkedArray)

    view_keep = TensorView(chunked, chunk_policy="keep")
    assert isinstance(view_keep._data, pa.ChunkedArray)

    view_combine = TensorView(chunked, chunk_policy="combine")
    assert isinstance(view_combine._data, pa.StructArray)


def test_tensor_view_chunk_policy_invalid(example_correct_data: dict) -> None:
    """Reject unsupported chunk policy values."""
    oa = OMEArrow(example_correct_data)

    with pytest.raises(ValueError, match="Unsupported chunk_policy"):
        oa.tensor_view(chunk_policy="invalid")


def test_lazy_tensor_view_collects_on_execution(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Defer source loading until lazy tensor view execution."""
    oa = OMEArrow.scan("tests/data/JUMP-BR00117006/BR00117006.ome.parquet")
    assert oa.is_lazy

    view = oa.tensor_view(t=0, z=0, c=0, layout="YX")
    assert isinstance(view, LazyTensorView)
    assert oa.is_lazy

    arr = view.to_numpy(contiguous=True)
    assert arr.shape == (72, 84)
    assert oa.is_lazy
    if recwarn:
        assert any(
            "Requested column 'ome_arrow'" in str(w.message) for w in recwarn.list
        )


def test_lazy_reader_requires_string_source() -> None:
    """Reject lazy mode for non-file inputs."""
    arr = np.zeros((1, 1, 1, 2, 2), dtype=np.uint16)
    with pytest.raises(TypeError, match="lazy=True currently supports only string"):
        OMEArrow(arr, lazy=True)


@pytest.mark.filterwarnings(
    "ignore:As of version 0.4.0, the parser argument is ignored.*:DeprecationWarning"
)
def test_lazy_tensor_view_select_preserves_existing_dims() -> None:
    """Preserve existing lazy selections when select() updates one axis."""
    oa = OMEArrow.scan(
        "tests/data/ome-artificial-5d-datasets/multi-channel-time-series.ome.tiff"
    )
    assert oa.is_lazy

    view = oa.tensor_view(t=2, c=1)
    assert isinstance(view, LazyTensorView)

    view_z = view.select(z=0)
    assert isinstance(view_z, LazyTensorView)
    assert oa.is_lazy

    arr = view_z.to_numpy(contiguous=True)
    assert arr.shape == (1, 167, 439)
    assert oa.is_lazy


def test_lazy_tensor_view_with_layout_defers_materialization() -> None:
    """Update layout lazily and materialize only on execution."""
    oa = OMEArrow.scan("tests/data/JUMP-BR00117006/BR00117006.ome.parquet")
    assert oa.is_lazy

    view = oa.tensor_view(t=0, z=0, c=0)
    view_yx = view.with_layout("YX")

    assert isinstance(view_yx, LazyTensorView)
    assert view_yx.layout == "YX"
    assert oa.is_lazy

    with pytest.warns(UserWarning, match="Requested column 'ome_arrow'"):
        concrete = view_yx.collect()
    assert concrete.layout in {"YX", "HW"}
    arr = concrete.to_numpy(contiguous=True)
    assert arr.shape == (72, 84)
    assert oa.is_lazy


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


def test_tensor_view_layout_yx_with_first_channel_policy(
    example_correct_data: dict,
) -> None:
    """Allow YX layout by selecting the first channel when requested."""
    oa = OMEArrow(example_correct_data)

    view_yx = oa.tensor_view(t=0, z=0, layout="YX", channel_policy="first")
    arr_yx = view_yx.to_numpy(contiguous=False)

    expected_cyx = oa.tensor_view(t=0, z=0, layout="CYX").to_numpy(contiguous=True)
    np.testing.assert_array_equal(arr_yx, expected_cyx[0])
    assert arr_yx.shape == (3, 4)


def test_tensor_view_roi3d_selects_z_and_roi() -> None:
    """Map roi3d to z slice + spatial roi consistently."""
    arr = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    oa = OMEArrow(arr)

    view = oa.tensor_view(roi3d=(1, 1, 1, 3, 2, 2), layout="TZCYX")
    out = view.to_numpy(contiguous=True)

    expected = arr[:, :, 1:3, 1:3, 1:4]
    expected = np.transpose(expected, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(out, expected)
    assert out.shape == (1, 2, 1, 2, 3)


def test_tensor_view_roi3d_conflicts_with_z() -> None:
    """Reject ambiguous selection when both roi3d and z are provided."""
    arr = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    oa = OMEArrow(arr)

    with pytest.raises(ValueError, match="Provide either z or roi3d"):
        oa.tensor_view(z=0, roi3d=(0, 0, 0, 2, 2, 1))


def test_tensor_view_roi_nd_3d_selects_z_and_roi() -> None:
    """Support roi_nd 3D bounds with explicit roi_type."""
    arr = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    oa = OMEArrow(arr)

    view = oa.tensor_view(roi_nd=(1, 1, 1, 3, 3, 4), roi_type="3d", layout="TZCYX")
    out = view.to_numpy(contiguous=True)

    expected = arr[:, :, 1:3, 1:3, 1:4]
    expected = np.transpose(expected, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(out, expected)
    assert out.shape == (1, 2, 1, 2, 3)


def test_tensor_view_roi_nd_2d_timelapse_selects_t_and_roi() -> None:
    """Support roi_nd timelapse bounds with explicit roi_type."""
    arr = np.arange(3 * 1 * 1 * 4 * 5, dtype=np.uint16).reshape(3, 1, 1, 4, 5)
    oa = OMEArrow(arr)

    view = oa.tensor_view(
        roi_nd=(1, 1, 1, 3, 3, 4), roi_type="2d_timelapse", layout="TZCYX"
    )
    out = view.to_numpy(contiguous=True)

    expected = arr[1:3, :, :, 1:3, 1:4]
    expected = np.transpose(expected, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(out, expected)
    assert out.shape == (2, 1, 1, 2, 3)


def test_tensor_view_roi_nd_4d_selects_t_z_and_roi() -> None:
    """Support roi_nd 4D bounds with implicit roi_type by tuple length."""
    arr = np.arange(3 * 1 * 4 * 5 * 6, dtype=np.uint16).reshape(3, 1, 4, 5, 6)
    oa = OMEArrow(arr)

    view = oa.tensor_view(roi_nd=(1, 1, 1, 2, 3, 3, 4, 5), layout="TZCYX")
    out = view.to_numpy(contiguous=True)

    expected = arr[1:3, :, 1:3, 1:4, 2:5]
    expected = np.transpose(expected, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(out, expected)
    assert out.shape == (2, 2, 1, 3, 3)


def test_tensor_view_roi_nd_len6_requires_roi_type() -> None:
    """Reject ambiguous roi_nd tuples with 6 values unless roi_type is set."""
    arr = np.arange(1 * 1 * 3 * 4 * 5, dtype=np.uint16).reshape(1, 1, 3, 4, 5)
    oa = OMEArrow(arr)

    with pytest.raises(ValueError, match="roi_nd with 6 values is ambiguous"):
        oa.tensor_view(roi_nd=(0, 0, 0, 1, 2, 3))


def test_dlpack_roundtrip_jax(example_correct_data: dict) -> None:
    """Round-trip DLPack export/import through JAX on CPU."""
    jnp = pytest.importorskip("jax.numpy")

    oa = OMEArrow(example_correct_data)
    view = oa.tensor_view(t=0, z=0)
    dlpack = view.to_dlpack(contiguous=True, mode="numpy")
    arr = jnp.from_dlpack(dlpack)

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

    with pytest.warns(
        UserWarning,
        match="mode='arrow' received a StructScalar; converting via as_py()",
    ):
        dlpack = view.to_dlpack(mode="arrow")
    arr = _from_dlpack_capsule(dlpack)
    expected = np.array(example_correct_data["planes"][0]["pixels"])
    np.testing.assert_array_equal(arr, expected)


def test_layout_drop_non_singleton_errors() -> None:
    """Reject layout drops when the omitted axis is non-singleton."""
    arr = np.zeros((2, 1, 1, 2, 2), dtype=np.uint16)
    oa = OMEArrow(arr)
    view = oa.tensor_view(layout="CYX")
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

    caps = list(view.iter_dlpack(tile_size=(2, 2), mode="numpy"))
    assert len(caps) == 4

    tile = _from_dlpack_capsule(caps[0])
    assert tile.shape == (2, 2, 2)


def test_iter_tiles_3d_numpy() -> None:
    """Yield expected 3D tile count and values."""
    arr = np.arange(1 * 1 * 3 * 4 * 4, dtype=np.uint16).reshape(1, 1, 3, 4, 4)
    oa = OMEArrow(arr)
    view = oa.tensor_view()

    caps = list(view.iter_tiles_3d(tile_size=(2, 2, 2), mode="numpy"))
    assert len(caps) == 8

    first = _from_dlpack_capsule(caps[0])
    assert first.shape == (1, 2, 1, 2, 2)
    expected = arr[:, :, 0:2, 0:2, 0:2]
    expected = np.transpose(expected, (0, 2, 1, 3, 4))
    np.testing.assert_array_equal(first, expected)


def test_iter_tiles_3d_arrow_mode_errors() -> None:
    """Reject arrow mode for 3D tiled iteration."""
    arr = np.arange(1 * 1 * 2 * 2 * 2, dtype=np.uint16).reshape(1, 1, 2, 2, 2)
    oa = OMEArrow(arr)

    with pytest.raises(ValueError, match="supports only mode='numpy'"):
        list(oa.tensor_view().iter_tiles_3d(tile_size=(1, 1, 1), mode="arrow"))


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
    values = _selected_values_for_arrow_mode(struct_arr, t=0, z=0, c=0)

    offset_bytes = values.offset * (values.type.bit_width // 8)
    ptr_arrow = values.buffers()[1].address + offset_bytes
    ptr_torch = torch.utils.dlpack.from_dlpack(capsule).data_ptr()
    if ptr_torch != ptr_arrow:
        pytest.skip("Torch did not preserve zero-copy for pyarrow DLPack.")


def test_arrow_mode_zero_copy_parquet_jax(
    tmp_path: pathlib.Path, example_correct_data: dict
) -> None:
    """Best-effort pointer check for JAX zero-copy from parquet."""
    jnp = pytest.importorskip("jax.numpy")

    out = tmp_path / "example.parquet"
    to_ome_parquet(example_correct_data, out_path=str(out), column_name="ome_arrow")

    oa = OMEArrow(str(out))
    view = oa.tensor_view(t=0, z=0, c=0)
    capsule = view.to_dlpack(mode="arrow")

    arr = jnp.from_dlpack(capsule)
    try:
        device = arr.device()
        assert device.platform == "cpu"
    except Exception:
        pytest.skip("Unable to read JAX device platform.")

    struct_arr = oa._struct_array
    assert struct_arr is not None
    values = _selected_values_for_arrow_mode(struct_arr, t=0, z=0, c=0)

    offset_bytes = values.offset * (values.type.bit_width // 8)
    ptr_arrow = values.buffers()[1].address + offset_bytes
    ptr_jax = _jax_buffer_ptr(arr)
    if ptr_jax != ptr_arrow:
        pytest.skip("JAX did not preserve zero-copy for pyarrow DLPack.")


def test_scan_tensor_view_to_numpy_avoids_python_record_materialization(
    tmp_path: pathlib.Path,
    example_correct_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read selected planes from Arrow without building a full Python record."""
    out = tmp_path / "lazy_arrow_plane.parquet"
    record = dict(example_correct_data)
    record["chunk_grid"] = None
    record["chunks"] = []
    to_ome_parquet(record, out_path=str(out), column_name="ome_arrow")

    oa = OMEArrow.scan(str(out))

    def _fail_data_py_dict(self: TensorView) -> dict:
        raise AssertionError("_data_py_dict should not be used for Arrow plane reads")

    monkeypatch.setattr(TensorView, "_data_py_dict", _fail_data_py_dict)
    arr = oa.tensor_view(t=0, z=0, c=0, layout="YX").to_numpy(contiguous=True)

    assert arr.shape == (3, 4)
    assert oa.is_lazy


def test_scan_chunked_parquet_tensor_view_avoids_python_record_materialization(
    tmp_path: pathlib.Path,
    example_correct_data: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read chunked parquet planes through Arrow path without _data_py_dict."""
    out = tmp_path / "lazy_chunked.parquet"
    to_ome_parquet(example_correct_data, out_path=str(out), column_name="ome_arrow")

    oa = OMEArrow.scan(str(out))

    def _fail_data_py_dict(self: TensorView) -> dict:
        raise AssertionError(
            "_data_py_dict should not be used for Arrow chunked plane reads"
        )

    monkeypatch.setattr(TensorView, "_data_py_dict", _fail_data_py_dict)
    arr = oa.tensor_view(t=0, z=0, c=1, layout="YX").to_numpy(contiguous=True)

    expected = np.array(
        [[100, 101, 102, 103], [110, 111, 112, 113], [120, 121, 122, 123]],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(arr, expected)
    assert oa.is_lazy


@pytest.mark.filterwarnings(
    "ignore:As of version 0.4.0, the parser argument is ignored.*:DeprecationWarning"
)
def test_scan_tiff_tensor_view_uses_source_plane_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute lazy TIFF tensor views without calling eager from_tiff ingestion."""
    path = "tests/data/ome-artificial-5d-datasets/single-channel.ome.tiff"
    oa = OMEArrow.scan(path)

    def _fail_from_tiff(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("from_tiff should not be used for lazy tensor execution")

    monkeypatch.setattr(core_module, "from_tiff", _fail_from_tiff)
    arr = oa.tensor_view(t=0, z=0, c=0, layout="YX").to_numpy(contiguous=True)

    assert arr.shape == (167, 439)
    assert oa.is_lazy


def _jax_buffer_ptr(arr: object) -> int:
    """Return a best-effort device pointer for a JAX array.

    Notes:
        This probes version-specific JAX internals. Keep this helper local to
        tests and prefer skipping on unknown layouts over hard failures.
    """
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


def _selected_values_for_arrow_mode(
    struct_arr: object, *, t: int, z: int, c: int
) -> object:
    """Select Arrow pixel values for the exact path used by ``mode='arrow'``.

    Notes:
        This intentionally reimplements selection logic from production code so
        tests can independently verify behavior and pointer provenance.
    """
    chunks_arr = struct_arr.field("chunks")
    has_chunks = len(chunks_arr) > 0 and not chunks_arr.is_null().to_pylist()[0]
    if has_chunks:
        chunks = chunks_arr[0].values
        mask = pc.and_(
            pc.equal(chunks.field("t"), t),
            pc.and_(
                pc.equal(chunks.field("z"), z),
                pc.and_(
                    pc.equal(chunks.field("c"), c),
                    pc.and_(
                        pc.equal(chunks.field("x"), 0), pc.equal(chunks.field("y"), 0)
                    ),
                ),
            ),
        )
        selected = pc.filter(chunks, mask)
        return selected.field("pixels")[0].values

    planes = struct_arr.field("planes")[0].values
    mask = pc.and_(
        pc.equal(planes.field("t"), t),
        pc.and_(pc.equal(planes.field("z"), z), pc.equal(planes.field("c"), c)),
    )
    selected = pc.filter(planes, mask)
    return selected.field("pixels")[0].values
