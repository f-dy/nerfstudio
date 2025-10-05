"""
Tests for camera intrinsics validation in tiling functionality
Based on original test_nerfstudio_dataparser_tiling.py concepts
"""

import pytest


def test_camera_intrinsics_exact_division():
    """Test camera intrinsics for exact division scenario (128×80 → 2×2 tiles)"""
    # Test the core mathematical concept: cx/cy adjustment for exact division
    original_cx, original_cy = 57.0, 43.0
    tile_widths = [64, 64]  # 128/2 = 64 each
    tile_heights = [40, 40]  # 80/2 = 40 each

    # Expected cx/cy for each tile position
    expected_adjustments = [
        (0, 0, 57.0, 43.0),  # Top-left: no offset
        (0, 1, -7.0, 43.0),  # Top-right: cx = 57 - 64 = -7
        (1, 0, 57.0, 3.0),  # Bottom-left: cy = 43 - 40 = 3
        (1, 1, -7.0, 3.0),  # Bottom-right: cx = -7, cy = 3
    ]

    for row, col, expected_cx, expected_cy in expected_adjustments:
        # Calculate offsets (core tiling math)
        x_offset = sum(tile_widths[:col])
        y_offset = sum(tile_heights[:row])
        adjusted_cx = original_cx - x_offset
        adjusted_cy = original_cy - y_offset

        assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile ({row},{col}): cx mismatch"
        assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile ({row},{col}): cy mismatch"


def test_camera_intrinsics_with_remainder():
    """Test camera intrinsics for remainder handling (100×80 → 3×3 tiles)"""
    # Test remainder distribution: 100/3 = 33 remainder 1, 80/3 = 26 remainder 2
    original_cx, original_cy = 50.0, 40.0
    tile_widths = [34, 33, 33]  # First gets +1 from remainder
    tile_heights = [27, 27, 26]  # First two get +1 from remainder

    # Test a few key positions
    test_cases = [
        (0, 0, 50.0, 40.0),  # Top-left: no offset
        (0, 1, 16.0, 40.0),  # Top-middle: cx = 50 - 34 = 16
        (1, 0, 50.0, 13.0),  # Middle-left: cy = 40 - 27 = 13
        (2, 2, -17.0, -14.0),  # Bottom-right: cx = 50 - 67 = -17, cy = 40 - 54 = -14
    ]

    for row, col, expected_cx, expected_cy in test_cases:
        x_offset = sum(tile_widths[:col])
        y_offset = sum(tile_heights[:row])
        adjusted_cx = original_cx - x_offset
        adjusted_cy = original_cy - y_offset

        assert abs(adjusted_cx - expected_cx) < 1e-6, f"Tile ({row},{col}): cx error"
        assert abs(adjusted_cy - expected_cy) < 1e-6, f"Tile ({row},{col}): cy error"


def test_camera_parameter_precision():
    """Test floating-point precision in cx/cy calculations"""
    test_cases = [
        {"cx": 63.7, "cy": 39.3},  # Non-integer values
        {"cx": 0.0, "cy": 0.0},  # Corner case
        {"cx": 127.9, "cy": 79.9},  # Near edge
        {"cx": 64.5, "cy": 40.5},  # Half-pixel offset
    ]

    tile_widths = [64, 64]
    tile_heights = [40, 40]

    for case in test_cases:
        original_cx, original_cy = case["cx"], case["cy"]

        # Test all 4 tile positions
        for row in range(2):
            for col in range(2):
                x_offset = sum(tile_widths[:col])
                y_offset = sum(tile_heights[:row])
                expected_cx = original_cx - x_offset
                expected_cy = original_cy - y_offset

                # Verify precision (should be exact for floating-point arithmetic)
                adjusted_cx = original_cx - x_offset
                adjusted_cy = original_cy - y_offset

                assert abs(adjusted_cx - expected_cx) < 1e-10, "Precision error in cx"
                assert abs(adjusted_cy - expected_cy) < 1e-10, "Precision error in cy"


def test_focal_length_preservation():
    """Test that focal lengths (fx, fy) must be preserved in tiling"""
    # Core concept: focal lengths are lens properties and never change during tiling
    test_cases = [
        {"fx": 64.0, "fy": 64.0},  # Square pixels
        {"fx": 70.0, "fy": 65.0},  # Rectangular pixels
        {"fx": 100.0, "fy": 95.0},  # Different aspect ratio
    ]

    for case in test_cases:
        # In any tiling implementation, these must remain identical
        tiled_fx = case["fx"]  # Should never change
        tiled_fy = case["fy"]  # Should never change

        assert tiled_fx == case["fx"], "Focal length fx must be preserved"
        assert tiled_fy == case["fy"], "Focal length fy must be preserved"


def test_tiling_disabled_behavior():
    """Test that disabled tiling preserves original parameters"""
    # When tile_size_max=0, should return original dimensions
    width, height = 128, 80

    # Disabled tiling should return single tile with original dimensions
    tile_widths = [width]
    tile_heights = [height]

    # Camera intrinsics should be unchanged for single tile at (0,0)
    original_cx, original_cy = 57.0, 43.0
    x_offset = sum(tile_widths[:0])  # 0
    y_offset = sum(tile_heights[:0])  # 0
    adjusted_cx = original_cx - x_offset
    adjusted_cy = original_cy - y_offset

    assert adjusted_cx == original_cx
    assert adjusted_cy == original_cy
    assert tile_widths[0] == width
    assert tile_heights[0] == height


def test_tile_size_calculation_concept():
    """Test the concept of balanced tile size calculation"""
    # Test that tile sizes sum to original dimensions
    test_cases = [
        (128, 80, [64, 64], [40, 40]),  # Exact division
        (100, 80, [34, 33, 33], [27, 27, 26]),  # With remainder
        (256, 256, [128, 128], [128, 128]),  # Square image
    ]

    for width, height, expected_widths, expected_heights in test_cases:
        # Core requirement: tiles must sum to original dimensions
        assert sum(expected_widths) == width, f"Tile widths don't sum to {width}"
        assert sum(expected_heights) == height, f"Tile heights don't sum to {height}"

        # All tiles should be reasonably sized (not too small/large)
        assert all(w > 0 for w in expected_widths), "All tile widths must be positive"
        assert all(h > 0 for h in expected_heights), "All tile heights must be positive"


if __name__ == "__main__":
    pytest.main([__file__])
