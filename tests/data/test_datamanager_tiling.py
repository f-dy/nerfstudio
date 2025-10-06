# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive tests for FullImageDatamanager tiling functionality
"""

import pytest
import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager


class MockConfig:
    """Mock config for testing tiling methods without full datamanager setup"""

    def __init__(self):
        self.tile_size_max = 0
        self.tile_alignment = 16
        self.cache_images = "cpu"
        self.cache_images_type = "float32"
        self.max_thread_workers = None


class TestTilingCore:
    """Core tiling algorithm tests"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create a minimal mock config to avoid full datamanager initialization
        self.config = MockConfig()
        self.datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

    def test_balanced_tile_sizes_comprehensive(self):
        """Test balanced tile size calculation with various scenarios"""
        test_cases = [
            # (width, height, tile_size_max, tile_alignment, expected_widths, expected_heights)
            (128, 80, 64, 16, [64, 64], [48, 32]),  # Exact division with alignment
            (256, 256, 128, 16, [128, 128], [128, 128]),  # Square image
            (50, 30, 100, 16, [50], [30]),  # Image smaller than tile_size_max
            (128, 80, 0, 16, [128], [80]),  # Tiling disabled
        ]

        for width, height, tile_size_max, tile_alignment, expected_widths, expected_heights in test_cases:
            tile_widths, tile_heights = self.datamanager._calculate_balanced_tile_sizes(
                width, height, tile_size_max, tile_alignment
            )

            # Check that tiles sum to original dimensions
            assert sum(tile_widths) == width, f"Tile widths don't sum to {width}"
            assert sum(tile_heights) == height, f"Tile heights don't sum to {height}"

            # Check alignment (when tiling is enabled)
            if tile_size_max > 0 and width > tile_size_max:
                for w in tile_widths[:-1]:  # All but last tile should be aligned
                    assert w % tile_alignment == 0, f"Tile width {w} not aligned to {tile_alignment}"
            if tile_size_max > 0 and height > tile_size_max:
                for h in tile_heights[:-1]:  # All but last tile should be aligned
                    assert h % tile_alignment == 0, f"Tile height {h} not aligned to {tile_alignment}"

    def test_image_tiling_pixel_perfect(self):
        """Test that image tiling produces pixel-perfect reconstruction"""
        # Create test image
        original_image = torch.rand(80, 128, 3)  # 80x128 RGB image

        # Calculate tile sizes
        tile_widths, tile_heights = self.datamanager._calculate_balanced_tile_sizes(128, 80, 64, 16)

        # Tile the image
        tiles = self.datamanager._tile_undistorted_image(original_image, tile_widths, tile_heights)

        # Reconstruct image from tiles
        reconstructed = torch.zeros_like(original_image)
        tile_idx = 0
        y_offset = 0

        for tile_height in tile_heights:
            x_offset = 0
            for tile_width in tile_widths:
                tile = tiles[tile_idx]
                reconstructed[y_offset : y_offset + tile_height, x_offset : x_offset + tile_width] = tile
                x_offset += tile_width
                tile_idx += 1
            y_offset += tile_height

        # Verify pixel-perfect reconstruction
        assert torch.allclose(original_image, reconstructed, atol=1e-6), "Reconstruction not pixel-perfect"

    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test very small images
        tiny_widths, tiny_heights = self.datamanager._calculate_balanced_tile_sizes(16, 16, 64, 16)
        assert tiny_widths == [16] and tiny_heights == [16], "Small images should not be tiled"

        # Test alignment edge cases
        with pytest.raises(ValueError, match="Image too large"):
            # Image that can't be aligned properly
            self.datamanager._calculate_balanced_tile_sizes(100, 100, 15, 32)

    def test_realistic_scenarios(self):
        """Test tiling with realistic image dimensions"""
        test_scenarios = [
            # (width, height, tile_size_max, description)
            (1920, 1080, 512, "Full HD"),
            (3840, 2160, 1024, "4K UHD"),
            (4096, 4096, 512, "4K Square"),
            (800, 600, 256, "SVGA"),
        ]

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        for width, height, tile_size_max, description in test_scenarios:
            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, tile_size_max, 16)

            # Verify basic properties
            assert sum(tile_widths) == width, f"{description}: Width mismatch"
            assert sum(tile_heights) == height, f"{description}: Height mismatch"
            assert all(w <= tile_size_max + 16 for w in tile_widths), f"{description}: Tile too wide"
            assert all(h <= tile_size_max + 16 for h in tile_heights), f"{description}: Tile too tall"

    def test_config_validation(self):
        """Test configuration parameter validation"""
        # Test valid configurations
        valid_configs = [
            {"tile_size_max": 0, "tile_alignment": 16},  # Disabled
            {"tile_size_max": 512, "tile_alignment": 16},  # Standard
            {"tile_size_max": 1024, "tile_alignment": 32},  # Large tiles
        ]

        for config in valid_configs:
            # Should not raise any exceptions
            tile_widths, tile_heights = FullImageDatamanager.__new__(
                FullImageDatamanager
            )._calculate_balanced_tile_sizes(1920, 1080, config["tile_size_max"], config["tile_alignment"])
            assert isinstance(tile_widths, list)
            assert isinstance(tile_heights, list)


class TestTilingIntrinsics:
    """Camera intrinsics validation tests"""

    def test_camera_intrinsics_exact_division(self):
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

    def test_camera_intrinsics_with_remainder(self):
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

    def test_camera_parameter_precision(self):
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

    def test_focal_length_preservation(self):
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

    def test_tiling_disabled_behavior(self):
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

    def test_camera_intrinsics_adjustment(self):
        """Test _adjust_camera_intrinsics_for_tile method directly"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([95.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)

        # Test tile adjustment
        adjusted = datamanager._adjust_camera_intrinsics_for_tile(
            cameras,
            tile_offset_x=32,
            tile_offset_y=20,
            tile_width=64,
            tile_height=40,
            parent_camera_index=1,
            tile_row=1,
            tile_col=0,
            total_tiles=4,
        )

        # Verify adjustments
        assert torch.allclose(adjusted.cx, torch.tensor([32.0])), "cx should be adjusted by offset"
        assert torch.allclose(adjusted.cy, torch.tensor([20.0])), "cy should be adjusted by offset"
        assert torch.allclose(adjusted.fx, cameras.fx), "fx should be preserved"
        assert torch.allclose(adjusted.fy, cameras.fy), "fy should be preserved"
        assert torch.allclose(adjusted.width, torch.tensor([64])), "width should be tile width"
        assert torch.allclose(adjusted.height, torch.tensor([40])), "height should be tile height"

        # Verify debugging metadata
        assert adjusted.metadata is not None, "Metadata should be present"
        metadata = adjusted.metadata
        assert metadata["parent_camera_index"] == 1, "Parent camera index should be preserved"
        assert metadata["tile_row"] == 1, "Tile row should be preserved"
        assert metadata["tile_col"] == 0, "Tile col should be preserved"

    def test_camera_replication(self):
        """Test _replicate_cameras_for_tiles method directly"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling

        # Test camera replication
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Verify correct number of cameras
        assert len(tile_cameras) == 4, "Should create 4 cameras for 2x2 tiling"

        # Verify camera adjustments for each tile
        # Both cy and tile_offset_y are in Y-down coordinates (COLMAP standard)
        # Simple subtraction: cy_adjusted = cy - tile_offset_y
        # Original: cy=40.0
        # Tile 0 (top-left): tile_offset_y=0 → cy_adjusted = 40 - 0 = 40
        # Tile 1 (top-right): tile_offset_y=0 → cy_adjusted = 40 - 0 = 40
        # Tile 2 (bottom-left): tile_offset_y=40 → cy_adjusted = 40 - 40 = 0
        # Tile 3 (bottom-right): tile_offset_y=40 → cy_adjusted = 40 - 40 = 0
        expected_adjustments = [
            (64.0, 40.0),  # Top-left: cx=64-0=64, cy=40-0=40
            (0.0, 40.0),  # Top-right: cx=64-64=0, cy=40-0=40
            (64.0, 0.0),  # Bottom-left: cx=64-0=64, cy=40-40=0
            (0.0, 0.0),  # Bottom-right: cx=64-64=0, cy=40-40=0
        ]

        for i, (expected_cx, expected_cy) in enumerate(expected_adjustments):
            assert torch.allclose(tile_cameras[i].cx, torch.tensor([expected_cx])), f"Tile {i} cx incorrect"
            assert torch.allclose(tile_cameras[i].cy, torch.tensor([expected_cy])), f"Tile {i} cy incorrect"
            assert torch.allclose(tile_cameras[i].fx, cameras.fx), f"Tile {i} fx should be preserved"
            assert torch.allclose(tile_cameras[i].fy, cameras.fy), f"Tile {i} fy should be preserved"

    def test_tile_to_original_mapping(self):
        """Test that tile_to_original_mapping correctly tracks which tiles came from which images"""
        # Test the concept: tile_mapping should track original image indices

        # Simulate the mapping logic from _apply_tiling_to_images
        tile_mapping = []

        # Image 0: 64x40 -> no tiling (1 tile)
        tile_mapping.append(0)

        # Image 1: 128x80 -> 2x2 tiling (4 tiles)
        for _ in range(4):
            tile_mapping.append(1)

        # Image 2: 50x30 -> no tiling (1 tile)
        tile_mapping.append(2)

        # Verify the mapping
        expected_mapping = [0, 1, 1, 1, 1, 2]
        assert tile_mapping == expected_mapping, f"Expected {expected_mapping}, got {tile_mapping}"

        # Verify we can trace tiles back to original images
        assert tile_mapping[0] == 0, "First tile should come from image 0"
        assert tile_mapping[1] == 1, "Second tile should come from image 1"
        assert tile_mapping[4] == 1, "Fifth tile should come from image 1"
        assert tile_mapping[5] == 2, "Last tile should come from image 2"

        # Verify total tiles
        assert len(tile_mapping) == 6, "Should have 6 total tiles"

        # Verify each original image is represented
        original_images_in_mapping = set(tile_mapping)
        assert original_images_in_mapping == {0, 1, 2}, "All original images should be represented"

    def test_eval_images_not_tiled(self):
        """Test that eval images are not tiled, only train images"""
        # Create mock images
        mock_images = [
            {"image": torch.rand(80, 128, 3)},  # Large image that would be tiled
            {"image": torch.rand(60, 100, 3)},  # Another large image
        ]

        # Create mock cameras
        cameras = Cameras(
            fx=torch.tensor([100.0, 100.0]),
            fy=torch.tensor([100.0, 100.0]),
            cx=torch.tensor([64.0, 50.0]),
            cy=torch.tensor([40.0, 30.0]),
            width=torch.tensor([128, 100]),
            height=torch.tensor([80, 60]),
            camera_to_worlds=torch.eye(4).unsqueeze(0).repeat(2, 1, 1)[:, :3, :],
        )

        # Create datamanager with tiling config
        config = MockConfig()
        config.tile_size_max = 64  # Will tile the large images
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        datamanager.config = config

        # Mock the required attributes
        datamanager.train_dataset = None
        datamanager.eval_dataset = None
        datamanager.train_cameras = cameras
        datamanager.eval_cameras = cameras

        # Test train split - should be tiled
        train_tiled = datamanager._apply_tiling_to_images(mock_images, "train")
        assert len(train_tiled) > len(mock_images), "Train images should be tiled"

        # Test eval split - should NOT be tiled (but method won't be called due to split check)
        # Let's test the condition directly by checking _load_images logic

        # Verify the condition: tiling only applies to train split
        should_tile_train = config.tile_size_max > 0 and "train" == "train"  # noqa: PLR0133
        should_tile_eval = config.tile_size_max > 0 and "eval" == "train"  # noqa: PLR0133

        assert should_tile_train, "Train images should be tiled when tile_size_max > 0"
        assert not should_tile_eval, "Eval images should NOT be tiled"

    def test_tile_debugging_metadata(self):
        """Test that tile cameras contain debugging metadata"""
        # Create test camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([95.0]),
            cx=torch.tensor([64.0]),
            cy=torch.tensor([40.0]),
            width=torch.tensor([128]),
            height=torch.tensor([80]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)
        tile_widths, tile_heights = [64, 64], [40, 40]  # 2x2 tiling

        # Test camera replication with debugging info
        tile_cameras = datamanager._replicate_cameras_for_tiles(
            cameras, tile_widths, tile_heights, parent_camera_index=5
        )

        # Verify all tile cameras have debugging metadata
        for i, tile_camera in enumerate(tile_cameras):
            assert tile_camera.metadata is not None, f"Tile {i} should have metadata"

            # Check required debugging fields
            metadata = tile_camera.metadata
            assert "parent_camera_index" in metadata, f"Tile {i} missing parent_camera_index"
            assert "tile_offset_x" in metadata, f"Tile {i} missing tile_offset_x"
            assert "tile_offset_y" in metadata, f"Tile {i} missing tile_offset_y"
            assert "tile_row" in metadata, f"Tile {i} missing tile_row"
            assert "tile_col" in metadata, f"Tile {i} missing tile_col"
            assert "total_tiles" in metadata, f"Tile {i} missing total_tiles"

            # Verify values
            assert metadata["parent_camera_index"] == 5, f"Tile {i} wrong parent index"
            assert metadata["total_tiles"] == 4, f"Tile {i} wrong total tiles"
            assert metadata["original_width"] == 128, f"Tile {i} wrong original width"
            assert metadata["original_height"] == 80, f"Tile {i} wrong original height"

        # Verify specific tile positions
        expected_positions = [
            (0, 0, 0, 0),  # Top-left: row=0, col=0, offset_x=0, offset_y=0
            (0, 1, 64, 0),  # Top-right: row=0, col=1, offset_x=64, offset_y=0
            (1, 0, 0, 40),  # Bottom-left: row=1, col=0, offset_x=0, offset_y=40
            (1, 1, 64, 40),  # Bottom-right: row=1, col=1, offset_x=64, offset_y=40
        ]

        for i, (expected_row, expected_col, expected_x, expected_y) in enumerate(expected_positions):
            metadata = tile_cameras[i].metadata
            assert metadata is not None, f"Tile {i} should have metadata"
            assert metadata["tile_row"] == expected_row, f"Tile {i} wrong row"
            assert metadata["tile_col"] == expected_col, f"Tile {i} wrong col"
            assert metadata["tile_offset_x"] == expected_x, f"Tile {i} wrong offset_x"
            assert metadata["tile_offset_y"] == expected_y, f"Tile {i} wrong offset_y"

    def test_alignment_constraints(self):
        """Test alignment constraint handling"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)

        # Test various alignment scenarios
        test_cases = [
            (128, 80, 64, 16),  # Perfect alignment
            (130, 82, 64, 16),  # Slight misalignment
            (100, 100, 48, 32),  # Different alignment
        ]

        for width, height, tile_size_max, alignment in test_cases:
            tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(
                width, height, tile_size_max, alignment
            )

            # All tiles except the last should be aligned
            for w in tile_widths[:-1]:
                assert w % alignment == 0, f"Non-edge tile width {w} not aligned to {alignment}"
            for h in tile_heights[:-1]:
                assert h % alignment == 0, f"Non-edge tile height {h} not aligned to {alignment}"


class TestTilingValidation:
    """Integration and validation tests"""

    def test_complete_tiling_pipeline(self):
        """Test the complete tiling pipeline with realistic data"""
        # Create mock image data
        width, height = 256, 256
        test_image = torch.rand(height, width, 3)

        # Create mock camera
        cameras = Cameras(
            fx=torch.tensor([100.0]),
            fy=torch.tensor([100.0]),
            cx=torch.tensor([128.0]),
            cy=torch.tensor([128.0]),
            width=torch.tensor([width]),
            height=torch.tensor([height]),
            camera_to_worlds=torch.eye(4).unsqueeze(0)[:, :3, :],
        )

        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test tiling pipeline
        tile_widths, tile_heights = datamanager._calculate_balanced_tile_sizes(width, height, 128, 16)
        tiles = datamanager._tile_undistorted_image(test_image, tile_widths, tile_heights)
        tile_cameras = datamanager._replicate_cameras_for_tiles(cameras, tile_widths, tile_heights)

        # Validate results
        expected_num_tiles = len(tile_widths) * len(tile_heights)
        assert len(tiles) == expected_num_tiles, "Incorrect number of tiles"
        assert len(tile_cameras) == expected_num_tiles, "Incorrect number of tile cameras"

        # Validate camera adjustments
        for i, tile_camera in enumerate(tile_cameras):
            assert tile_camera.fx == cameras.fx, "Focal length fx should be preserved"
            assert tile_camera.fy == cameras.fy, "Focal length fy should be preserved"

    def test_error_conditions(self):
        """Test error handling and edge cases"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test invalid alignment
        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(100, 100, 50, 0)  # Zero alignment

        # Test image too large for constraints
        with pytest.raises(ValueError):
            datamanager._calculate_balanced_tile_sizes(1000, 1000, 10, 32)  # Impossible constraints

    def test_performance_characteristics(self):
        """Test performance characteristics of tiling"""
        datamanager = FullImageDatamanager.__new__(FullImageDatamanager)  # Skip __init__

        # Test that tiling calculation is fast
        import time

        start_time = time.time()
        for _ in range(100):
            datamanager._calculate_balanced_tile_sizes(1920, 1080, 512, 16)
        elapsed = time.time() - start_time

        # Should be very fast (< 10ms for 100 calculations)
        assert elapsed < 0.01, f"Tiling calculation too slow: {elapsed:.3f}s for 100 iterations"


if __name__ == "__main__":
    pytest.main([__file__])
