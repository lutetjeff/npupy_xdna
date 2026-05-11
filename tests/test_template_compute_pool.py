from __future__ import annotations

import re

import numpy as np
import pytest

from npupy_xdna.regions.region import ArraySpec, Region
from npupy_xdna.templates.compute_pool import ComputePoolTemplate
from npupy_xdna.templates.shape_matrix import SUPPORTED_SHAPES


@pytest.fixture(scope="module")
def template():
    return ComputePoolTemplate()


class TestMatchGating:
    def test_matches_elementwise_unary_32768(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(32768,), dtype="int16")],
            output=ArraySpec(shape=(32768,), dtype="int16"),
        )
        assert template.match(r)

    def test_matches_elementwise_binary_32768(self, template):
        r = Region(
            op="elementwise_binary",
            inputs=[
                ArraySpec(shape=(32768,), dtype="int16"),
                ArraySpec(shape=(32768,), dtype="int16"),
            ],
            output=ArraySpec(shape=(32768,), dtype="int16"),
        )
        assert template.match(r)

    def test_rejects_size_below_minimum(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(16384,), dtype="int16")],
            output=ArraySpec(shape=(16384,), dtype="int16"),
        )
        assert not template.match(r)

    def test_rejects_size_not_in_supported_shapes(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(65536,), dtype="int16")],
            output=ArraySpec(shape=(65536,), dtype="int16"),
        )
        assert not template.match(r)

    def test_rejects_matmul_op(self, template):
        r = Region(
            op="matmul",
            inputs=[
                ArraySpec(shape=(32, 32), dtype="int16"),
                ArraySpec(shape=(32, 32), dtype="int16"),
            ],
            output=ArraySpec(shape=(32, 32), dtype="int16"),
        )
        assert not template.match(r)

    def test_matches_all_supported_sizes(self, template):
        for size in SUPPORTED_SHAPES["compute_pool"]:
            r = Region(
                op="elementwise_unary",
                inputs=[ArraySpec(shape=(size,), dtype="int16")],
                output=ArraySpec(shape=(size,), dtype="int16"),
            )
            assert template.match(r), f"should match size={size}"


class TestConfigSpace:
    def test_config_has_32_cores(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(32768,), dtype="int16")],
            output=ArraySpec(shape=(32768,), dtype="int16"),
        )
        configs = template.config_space(r)
        assert len(configs) == 1
        assert configs[0].n_cores == 32

    def test_chunk_per_core_correct(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(32768,), dtype="int16")],
            output=ArraySpec(shape=(32768,), dtype="int16"),
        )
        configs = template.config_space(r)
        assert configs[0].tile == (1024,), f"expected (1024,) got {configs[0].tile}"

    def test_chunk_scales_with_size(self, template):
        for size in SUPPORTED_SHAPES["compute_pool"]:
            r = Region(
                op="elementwise_unary",
                inputs=[ArraySpec(shape=(size,), dtype="int16")],
                output=ArraySpec(shape=(size,), dtype="int16"),
            )
            configs = template.config_space(r)
            expected_chunk = size // 32
            assert configs[0].tile == (expected_chunk,)


class TestMlirStructure:
    @pytest.fixture(scope="class")
    def mlir_str(self, template):
        r = Region(
            op="elementwise_unary",
            inputs=[ArraySpec(shape=(32768,), dtype="int16")],
            output=ArraySpec(shape=(32768,), dtype="int16"),
        )
        configs = template.config_space(r)
        iron_fn = template.lower(r, configs[0])
        return str(iron_fn())

    def test_targets_npu2(self, mlir_str):
        assert "aie.device(npu2)" in mlir_str

    def test_32_distinct_compute_tiles(self, mlir_str):
        tiles = re.findall(r"aie\.tile\((\d+),\s*([2-5])\)", mlir_str)
        unique_compute = set(tiles)
        assert len(unique_compute) == 32, (
            f"Expected 32 distinct compute tiles (col 0-7, row 2-5), found {len(unique_compute)}: {unique_compute}"
        )

    def test_no_broadcast_single_consumers(self, mlir_str):
        broadcast_pattern = re.findall(
            r"aie\.objectfifo\s+@\w+\([^,]+,\s*\{[^}]+,[^}]+\}", mlir_str
        )
        compute_tile_refs = [
            p for p in broadcast_pattern if "tile_" in p and "mem_tile" not in p and "shim" not in p
        ]
        assert len(compute_tile_refs) == 0, (
            f"Found objectfifo with multiple compute tile consumers (broadcast): {compute_tile_refs}"
        )

    def test_shape_32768_in_memref(self, mlir_str):
        assert "32768" in mlir_str

    def test_chunk_size_1024_in_memref(self, mlir_str):
        assert "1024" in mlir_str

    def test_link_with_distinct_offsets(self, mlir_str):
        links = re.findall(r"aie\.objectfifo\.link.*?->.*?\(.*?\[(.*?)\].*?\)", mlir_str)
        for link in links:
            offsets = [int(x.strip()) for x in link.split(",") if x.strip().isdigit()]
            if len(offsets) > 1:
                assert len(offsets) == len(set(offsets)), (
                    f"Duplicate offsets in objectfifo.link — this would be broadcast: {offsets}"
                )

    def test_8_memtiles_present(self, mlir_str):
        memtiles = re.findall(r"aie\.tile\((\d+),\s*1\)", mlir_str)
        unique_mem = set(memtiles)
        assert len(unique_mem) == 8, f"Expected 8 memtiles, found {len(unique_mem)}"

    def test_no_multicast_annotation(self, mlir_str):
        assert "multicast" not in mlir_str.lower()
        assert "broadcast" not in mlir_str.lower()

    def test_col_chunk_4096_in_memref(self, mlir_str):
        assert "4096" in mlir_str

    def test_template_name(self, template):
        assert template.name == "compute_pool"
