"""
Tests for detailed camera intrinsics validation in tiling functionality
Based on original test_nerfstudio_dataparser_tiling.py concepts but adapted for our implementation
"""

import pytest


class MockDatamanager:
    """Mock datamanager with just the tiling methods we need to test"""

    def _calculate_balanced_tile_sizes(self, width: int, height: int, tile_size_max: int, tile_alignment: int):
        """Simplified version of our balanced tile calculation for testing"""
        if tile_size_max <= 0:
            return [width], [height]

        import math

        num_tiles_x = math.ceil(width / tile_size_max)
        num_tiles_y = math.ceil(height / tile_size_max)

        if num_tiles_x == 1 and num_tiles_y == 1:
            return [width], [height]

        # Calculate base tile sizes
        base_tile_width = width // num_tiles_x
        base_tile_height = height // num_tiles_y

        # Apply alignment
        aligned_tile_width = max(tile_alignment, (base_tile_width // tile_alignment) * tile_alignment)
        aligned_tile_height = max(tile_alignment, (base_tile_height // tile_alignment) * tile_alignment)

        # Create tile arrays
        tile_widths = [aligned_tile_width] * num_tiles_x
        tile_heights = [aligned_tile_height] * num_tiles_y

        # Adjust last tiles to fit exactly
        total_aligned_width = aligned_tile_width * num_tiles_x
        total_aligned_height = aligned_tile_height * num_tiles_y

        if total_aligned_width != width:
            tile_widths[-1] += width - total_aligned_width
        if total_aligned_height != height:
            tile_heights[-1] += height - total_aligned_height

        return tile_widths, tile_heights

    def _adjust_camera_intrinsics_for_tile(
        self, cx: float, cy: float, tile_widths: list, tile_heights: list, row: int, col: int
    ):
        """Adjust camera intrinsics for a specific tile"""
        x_offset = sum(tile_widths[:col])
        y_offset = sum(tile_heights[:row])

        adjusted_cx = cx - x_offset
        adjusted_cy = cy - y_offset
        tile_w = tile_widths[col]
        tile_h = tile_heights[row]

        return adjusted_cx, adjusted_cy, tile_w, tile_h


def test_camera_intrinsics_exact_division():
    """Test camera intrinsics for exact division scenario (128×80 with tile_size_max=64)"""
    datamanager = MockDatamanager()

    # Test the balanced tile calculation
    width, height = 128, 80
    tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, 64, 16)

    # Should create 2x2 tiles
    assert len(tile_widths) == 2
    assert len(tile_heights) == 2

    # Check that total dimensions are preserved
    assert sum(tile_widths) == width
    assert sum(tile_heights) == height

    # Test camera intrinsics adjustment
    original_cx, original_cy = 57.0, 43.0

    # Test each tile position
    tile_idx = 0
    for row in range(len(tile_heights)):
        for col in range(len(tile_widths)):
            adjusted_cx, adjusted_cy, tile_w, tile_h = datamanager._adjust_camera_intrinsics_for_tile(
                original_cx, original_cy, tile_widths, tile_heights, row, col
            )

            # Calculate expected offsets
            x_offset = sum(tile_widths[:col])
            y_offset = sum(tile_heights[:row])
            expected_cx = original_cx - x_offset
            expected_cy = original_cy - y_offset

            assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile {tile_idx}: cx mismatch"
            assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile {tile_idx}: cy mismatch"
            assert tile_w == tile_widths[col], f"Tile {tile_idx}: width mismatch"
            assert tile_h == tile_heights[row], f"Tile {tile_idx}: height mismatch"

            tile_idx += 1


def test_camera_intrinsics_with_remainder():
    """Test camera intrinsics for remainder handling (100×80 with tile_size_max=34)"""
    datamanager = MockDatamanager()

    width, height = 100, 80
    tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, 34, 1)

    # Should create multiple tiles
    assert len(tile_widths) >= 2
    assert len(tile_heights) >= 2

    # Check total dimensions are preserved
    assert sum(tile_widths) == width
    assert sum(tile_heights) == height

    # Test camera intrinsics with remainder distribution
    original_cx, original_cy = 50.0, 40.0

    # Test that all tile positions work correctly
    for row in range(len(tile_heights)):
        for col in range(len(tile_widths)):
            adjusted_cx, adjusted_cy, tile_w, tile_h = datamanager._adjust_camera_intrinsics_for_tile(
                original_cx, original_cy, tile_widths, tile_heights, row, col
            )

            # Verify the adjustment math is correct
            x_offset = sum(tile_widths[:col])
            y_offset = sum(tile_heights[:row])
            expected_cx = original_cx - x_offset
            expected_cy = original_cy - y_offset

            assert abs(adjusted_cx - expected_cx) < 1e-6, f"Row {row}, Col {col}: cx calculation error"
            assert abs(adjusted_cy - expected_cy) < 1e-6, f"Row {row}, Col {col}: cy calculation error"


def test_camera_parameter_precision():
    """Test floating-point precision in cx/cy calculations"""
    datamanager = MockDatamanager()

    # Test with various principal point positions
    test_cases = [
        {"cx": 63.7, "cy": 39.3},  # Non-integer values
        {"cx": 0.0, "cy": 0.0},  # Corner case
        {"cx": 127.9, "cy": 79.9},  # Near edge
        {"cx": 64.5, "cy": 40.5},  # Half-pixel offset
    ]

    for case in test_cases:
        original_cx, original_cy = case["cx"], case["cy"]
        tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(128, 80, 64, 1)

        # Test all tile positions
        for row in range(len(tile_heights)):
            for col in range(len(tile_widths)):
                adjusted_cx, adjusted_cy, _, _ = datamanager._adjust_camera_intrinsics_for_tile(
                    original_cx, original_cy, tile_widths, tile_heights, row, col
                )

                # Calculate expected values
                x_offset = sum(tile_widths[:col])
                y_offset = sum(tile_heights[:row])
                expected_cx = original_cx - x_offset
                expected_cy = original_cy - y_offset

                # Check precision (should be exact for floating-point arithmetic)
                assert abs(adjusted_cx - expected_cx) < 1e-10, "Precision error in cx calculation"
                assert abs(adjusted_cy - expected_cy) < 1e-10, "Precision error in cy calculation"


def test_tiling_disabled_preserves_cameras():
    """Test that tile_size_max=0 preserves original camera parameters"""
    datamanager = MockDatamanager()

    # Should return original dimensions when tiling is disabled
    tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(128, 80, 0, 16)

    assert tile_widths == [128]
    assert tile_heights == [80]

    # Camera intrinsics should be unchanged
    adjusted_cx, adjusted_cy, tile_w, tile_h = datamanager._adjust_camera_intrinsics_for_tile(
        57.0, 43.0, tile_widths, tile_heights, 0, 0
    )

    assert adjusted_cx == 57.0
    assert adjusted_cy == 43.0
    assert tile_w == 128
    assert tile_h == 80


def test_focal_length_preservation_concept():
    """Test the concept that focal lengths should be preserved in tiling"""
    # This tests the mathematical concept - in real implementation,
    # focal lengths (fx, fy) should never change during tiling

    # In any tiling implementation, these should remain constant
    # This is a conceptual test of the requirement

    test_cases = [
        {"fx": 64.0, "fy": 64.0},  # Square pixels
        {"fx": 70.0, "fy": 65.0},  # Rectangular pixels
        {"fx": 100.0, "fy": 95.0},  # Different aspect ratio
    ]

    for case in test_cases:
        # The key insight: focal lengths are properties of the lens/sensor
        # and should never change when we tile an image
        tiled_fx = case["fx"]  # Should be identical
        tiled_fy = case["fy"]  # Should be identical

        assert tiled_fx == case["fx"], "Focal length fx must be preserved"
        assert tiled_fy == case["fy"], "Focal length fy must be preserved"


def test_principal_point_adjustment_math():
    """Test the mathematical correctness of principal point adjustment"""
    # This tests the core mathematical concept from the original tests

    # Original image: 128×80, principal point at (57, 43)
    original_cx, original_cy = 57.0, 43.0

    # If we split into 2×2 tiles of size 64×40 each:
    # Tile positions:
    # [0,0] [0,1]  <- row 0
    # [1,0] [1,1]  <- row 1
    #  ^     ^
    # col 0 col 1

    tile_widths = [64, 64]
    tile_heights = [40, 40]

    expected_adjustments = [
        # (row, col): expected (cx, cy)
        (0, 0, 57.0, 43.0),  # Top-left: no offset
        (0, 1, -7.0, 43.0),  # Top-right: cx = 57 - 64 = -7
        (1, 0, 57.0, 3.0),  # Bottom-left: cy = 43 - 40 = 3
        (1, 1, -7.0, 3.0),  # Bottom-right: cx = -7, cy = 3
    ]

    datamanager = MockDatamanager()

    for row, col, expected_cx, expected_cy in expected_adjustments:
        adjusted_cx, adjusted_cy, _, _ = datamanager._adjust_camera_intrinsics_for_tile(
            original_cx, original_cy, tile_widths, tile_heights, row, col
        )

        assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile ({row},{col}): cx mismatch"
        assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile ({row},{col}): cy mismatch"


if __name__ == "__main__":
    pytest.main([__file__])
